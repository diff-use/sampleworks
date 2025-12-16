import os


# RosettaCommons/foundry uses the `environs` package but I'm trying to avoid dependencies
should_check_nans = os.environ.get("NAN_CHECK", default="True")
should_check_nans = False if should_check_nans.lower() in ("false", "0") else True
