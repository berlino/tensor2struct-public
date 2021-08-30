from tensor2struct.models import encoder, batched_encoder
from tensor2struct.utils import registry


class CogsPreproc(encoder.EncPreproc):
    def add_item(self, item, section, validation_info):
        tokens = item.text.split()
        self.texts[section].append({"tokens": tokens})

        if section == "train":
            for token in tokens:
                self.vocab_builder.add_word(token)


@registry.register("encoder", "cogs_enc")
class CogsEncoder(batched_encoder.Encoder):
    batched = True
    Preproc = CogsPreproc

class CogsLatPerPreproc(encoder.EncPreproc):
    def add_item(self, item, section, validation_info):
        tokens = item.text.split()

        # remove the period mark
        if tokens[-1] == ".":
            tokens = tokens[:-1]

        self.texts[section].append({"tokens": tokens})

        if section == "train":
            for token in tokens:
                self.vocab_builder.add_word(token)
