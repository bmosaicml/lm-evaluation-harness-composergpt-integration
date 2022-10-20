import torch
from lm_eval.base import BaseLM
from tqdm import tqdm
import torch
import torch.nn.functional as F
from composer.core.precision import get_precision_context, Precision

from lm_eval import utils


class ComposerLLM(BaseLM):
    def __init__(
        self,
        model, # Can be any torch module whose forward expects a dict w/ keys ['input_ids', 'attention_mask']
        tokenizer, # Can be any tokenizer whose forward method returns a dict w/ keys ['input_ids', 'attention_mask']
        device,
        precision: str,
        batch_size=4,
    ):
        super().__init__()

        self.precision = precision

        assert isinstance(device, str)
       

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        # multithreading and batching
        self.batch_size_per_gpu = batch_size 


    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.model.cfg.max_seq_len
       
    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        truncation = True
        return self.tokenizer(string,
            truncation=truncation,
        )

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            with get_precision_context(self.precision):
                res = self.model(inps)
            return res[:, :, :self.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = x[1]['input_ids'] + x[2]['input_ids']

            # I believe we want the continuation to be masked so that the model can't use it
            masks = x[1]['attention_mask'] + x[2]['attention_mask']
            return -len(toks), (tuple(toks), tuple(masks))

        re_ord = utils.Reorderer(requests, _collate)
        for chunk in utils.chunks(
            tqdm(re_ord.get_reordered(), disable=disable_tqdm), self.batch_size
        ):
            inps = {}
            cont_toks_list = []
            inplens = []


          
            for _, context_enc, continuation_enc in chunk:
                # mask out the continuation so we can't attend to it!
                continuation_enc['attention_mask'] = [0]*len(continuation_enc['attention_mask'])
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #                  CTX      CONT
                # inp              0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model.forward    \               \
                # logits            1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks               4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                padding_token = {'input_ids': self.tokenizer.tokenizer.pad_token_id, 'attention_mask': 0}

                inp = {}
                inplen = 0
                for k in context_enc:
                    # [:-1] because logits for idx i correpond to the prediction for token at idx i+1
                    input = (context_enc[k] + continuation_enc[k])[-(self.max_length + 1) :][:-1]
                    if k == 'input_ids':
                        inplen = len(input)
                    
                    # this model expects padding
                    padded_input = input + [padding_token[k]]*(self.max_length - len(input))
                    inp[k] = torch.tensor(
                        padded_input,
                        dtype=torch.long,
                    ).to(self.device)

              
                cont = continuation_enc['input_ids']
                for k in inp:
                    if k not in inps:
                        inps[k] = []
                    inps[k].append(inp[k])
                cont_toks_list.append(cont)
                inplens.append(inplen)


            batched_inps = {k: torch.stack(inps[k], dim=0) for k in inps}
            
            outputs = self._model_call(batched_inps)
            
            multi_logits = F.log_softmax(
                outputs, dim=-1
            ).cpu()  # [batch, padding_length, vocab]

            for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                chunk, multi_logits, inps['input_ids'], inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                logits = logits[inplen - contlen : inplen].unsqueeze(
                    0
                )  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
                    0
                )  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return re_ord.get_original(res)


