import torch
import torch.utils.data

from tensor2struct.models import abstract_preproc
from tensor2struct.utils import registry

import logging

logger = logging.getLogger("tensor2struct")

class ZippedDataset(torch.utils.data.Dataset):
    def __init__(self, *components):
        assert len(components) >= 1
        lengths = [len(c) for c in components]
        assert all(
            lengths[0] == other for other in lengths[1:]
        ), "Lengths don't match: {}".format(lengths)
        self.components = components

    def __getitem__(self, idx):
        return tuple(c[idx] for c in self.components)

    def __len__(self):
        return len(self.components[0])

    def concat(self, other):
        assert isinstance(other, ZippedDataset)
        for i, comp in enumerate(self.components):
            self.components[i].extend(other.components[i])


class EncDecPreproc(abstract_preproc.AbstractPreproc):
    def __init__(self, encoder, decoder, encoder_preproc, decoder_preproc):
        super().__init__()

        self.enc_preproc = registry.lookup("encoder", encoder["name"]).Preproc(
            **encoder_preproc
        )
        self.dec_preproc = registry.lookup("decoder", decoder["name"]).Preproc(
            **decoder_preproc
        )

    def validate_item(self, item, section):
        enc_result, enc_info = self.enc_preproc.validate_item(item, section)
        dec_result, dec_info = self.dec_preproc.validate_item(item, section)

        return enc_result and dec_result, (enc_info, dec_info)

    def add_item(self, item, section, validation_info):
        enc_info, dec_info = validation_info
        self.enc_preproc.add_item(item, section, enc_info)
        self.dec_preproc.add_item(item, section, dec_info)

    def clear_items(self):
        self.enc_preproc.clear_items()
        self.dec_preproc.clear_items()

    def save(self):
        self.enc_preproc.save()
        self.dec_preproc.save()

    def load(self):
        self.enc_preproc.load()
        self.dec_preproc.load()

    def dataset(self, section):
        return ZippedDataset(
            self.enc_preproc.dataset(section), self.dec_preproc.dataset(section)
        )


@registry.register("model", "EncDecV2")
class SemiBatchedEncDecModel(torch.nn.Module):
    """
    Encoder is batched but decoder is unbatched, this is for Spider
    """

    Preproc = EncDecPreproc

    def __init__(self, preproc, device, encoder, decoder):
        super().__init__()
        self.preproc = preproc
        self.encoder = registry.construct(
            "encoder", encoder, device=device, preproc=preproc.enc_preproc
        )
        self.decoder = registry.construct(
            "decoder", decoder, device=device, preproc=preproc.dec_preproc
        )

        assert getattr(self.encoder, "batched")  # use batched enc by default

    def forward(self, *input_items, compute_loss=True, infer=False):
        """
        The only entry point. In this unbatched version, input_items is 
        the input_batch during training; it is a tuple (orig_item, preproc_item)
        during inference time.
        Args:
            input_items: if it is a list, then we infer that it is a batch
            of examples for training; if is not a single list, then we infer
            that it's in inference mode
        """
        ret_dic = {}
        if compute_loss:
            assert len(input_items) == 1  # it's a batched version
            loss = self._compute_loss_enc_batched(input_items[0])
            ret_dic["loss"] = loss

        if infer:
            assert len(input_items) == 2  # unbatched version of inference
            orig_item, preproc_item = input_items
            infer_dic = self.begin_inference(orig_item, preproc_item)
            ret_dic = {**ret_dic, **infer_dic}
        return ret_dic

    def _compute_loss_enc_batched(self, batch):
        """
        Default way of computing loss: enc returns a list, 
        dec process enc outputs sequentially
        """
        losses = []
        enc_states = self.encoder([enc_input for enc_input, dec_output in batch])

        for enc_state, (enc_input, dec_output) in zip(enc_states, batch):
            ret_dic = self.decoder(dec_output, enc_state)
            losses.append(ret_dic["loss"])
        return torch.mean(torch.stack(losses, dim=0), dim=0)

    def begin_inference(self, orig_item, preproc_item):
        enc_input, _ = preproc_item
        (enc_state,) = self.encoder([enc_input])
        return self.decoder(orig_item, enc_state, compute_loss=False, infer=True)

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def get_bert_parameters_legacy(self):
        bert_params = list(self.encoder.bert_model.parameters())
        assert len(bert_params) > 0
        return bert_params

    def get_bert_parameters(self):
        bert_params = []
        for name, _param in self.named_parameters():
            if "bert" in name:
                bert_params.append(_param)
        return bert_params

    def get_non_bert_parameters(self):
        non_bert_params = []
        bert_params = set(self.get_bert_parameters())
        for name, _param in self.named_parameters():
            if _param not in bert_params:
                # if "bert" not in name:
                non_bert_params.append(_param)
        return non_bert_params


@registry.register("model", "UnBatchedEncDec")
class UnBatchedEncDecModel(torch.nn.Module):
    """
    Both encoder and decoder are unbatched
    """

    Preproc = EncDecPreproc

    def forward(self, *input_items, compute_loss=True, infer=False):
        """
        Similar to SemiBatchedEncDec, in training mode, input_items is a list 
        of length 1, containing a training batch; in decoding mode input_items is
        a pair of orig_item and preproc_item
        """
        ret_dic = {}
        if compute_loss:
            assert len(input_items) == 1  # it's a batch of training data
            loss = self._compute_loss_enc_unbatched(input_items[0])
            ret_dic["loss"] = loss

        if infer:
            assert len(input_items) == 2  # unbatched version of inference
            orig_item, preproc_item = input_items
            infer_dic = self.begin_inference(orig_item, preproc_item)
            ret_dic = {**ret_dic, **infer_dic}
        return ret_dic

    def _compute_loss_unbatched(self, batch):
        losses = []
        for enc_input, dec_output in batch:
            enc_state = self.encoder(enc_input)
            ret_dic = self.decoder(dec_output, enc_state)
            losses.append(ret_dic["loss"])
        return torch.mean(torch.stack(losses, dim=0), dim=0)

    def begin_inference(self, orig_item, preproc_item):
        enc_input, _ = preproc_item
        enc_state = self.encoder(enc_input)
        return self.decoder(orig_item, enc_state, compute_loss=False, infer=True)


@registry.register("model", "EncDec")
class BatchedEncDecModel(SemiBatchedEncDecModel):
    Preproc = EncDecPreproc

    def __init__(self, preproc, device, encoder, decoder):
        super().__init__(preproc, device, encoder, decoder)
        assert getattr(self.encoder, "batched") and getattr(self.decoder, "batched")

    def forward(self, input_batch, compute_loss=True, infer=False):
        """
        The only entry point of encdec. In this batched version, training 
        and inference also takes a preprocessed input batch

        Args:
            input_batch: for training, it contains both input and output; for decoding,
                it only contains input
        """
        ret_dic = {}
        if compute_loss:
            loss = self.compute_loss_batched(input_batch)
            ret_dic["loss"] = loss
        if infer:
            infer_dic = self.begin_batched_inference(input_batch)
            ret_dic = {**ret_dic, **infer_dic}
        return ret_dic

    def compute_loss_batched(self, batch):
        enc_batch = [enc_input for enc_input, dec_output in batch]
        dec_batch = [dec_output for enc_input, dec_output in batch]
        enc_state = self.encoder(enc_batch)
        ret_dic = self.decoder(dec_batch, enc_state)

        # encoder might have some auxilary loss
        if getattr(enc_state, "enc_loss", None):
            return ret_dic["loss"] + enc_state.enc_loss
        else:
            return ret_dic["loss"]

    def begin_batched_inference(self, enc_batch):
        """
        Unlike UnbatchedEncDec, enc_batch does not contain orig_items now. This
        might need to be supported in the future
        """
        enc_state = self.encoder(enc_batch)
        return self.decoder(None, enc_state, compute_loss=False, infer=True)

    def begin_inference(self, orig_item, preproc_item):
        """
        This function will be used in unbatched inference methods, such 
        as unbatched beam search
        """
        enc_input, _ = preproc_item
        enc_state = self.encoder([enc_input])
        return self.decoder.begin_inference(orig_item, enc_state)