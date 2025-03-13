PAMRAM_SET = {}

"""Demo test"""
PAMRAM_SET["000"] = ("",)

"""Comparison of different transformations for INRs on DIV2K-test (Table 1) """
PAMRAM_SET["001"] = (
            "01_norm",
            "z_score",

            #"gamma_0.5",
            #"gamma_2.0",

            # "scale_0.5",
            "scale_1.0",
            "scale_2.0",

            "inverse",
            # "rpp"
            "box_cox",
            
            "sym_power",
            
        )
