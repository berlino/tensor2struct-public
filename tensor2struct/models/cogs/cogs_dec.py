from tensor2struct.models import decoder, batched_decoder
from tensor2struct.utils import registry, vocab


class CogsPreproc(decoder.DecoderPreproc):
    def add_item(self, item, section, validation_info):
        actions = item.code.split()

        if section == "train":
            for action in actions:
                self.vocab_builder.add_word(action)

        self.items[section].append({"actions": [vocab.BOS] + actions + [vocab.EOS]})


@registry.register("decoder", "cogs_lstm_dec")
class CogsDecoder(batched_decoder.Decoder):
    batched = True
    Preproc = CogsPreproc


@registry.register("decoder", "cogs_transformer_dec")
class CogsTransformerDeccoder(batched_decoder.TransformerDecoder):
    batched = True
    Preproc = CogsPreproc
