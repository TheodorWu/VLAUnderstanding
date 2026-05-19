def build_perturbation_type_key(config):
    perturbator_config = config.get("perturbator", config) # assumes perturbator config is either at the top level or under "perturbator" key
    key = ""
    for k, v in perturbator_config.items():
        if v:
            key += f"{k}_and_"
    if key:
        return key.removesuffix("_and_")
    else:
        return "none"
