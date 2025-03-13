from types import SimpleNamespace
import os

STORE_TRUE = None

DIV2K_TEST = "./data/div2k/test_data"
DIV2K_TRAIN = "./data/div2k/train_data"
KODAK = "./data/Kodak"
TEXT_TEST = "./data/text/test_data"

DEMO_IMG = os.path.join(DIV2K_TEST, "00.png")


class ParamManager(object):
    def __init__(self, **kw):
        self._tag = "exp"
        self.p = SimpleNamespace()
        self._exp = ""

        self._set_exp(**kw)

    def _set_default_parmas(self):
        self.p.model_type = "siren"
        self.p.input_path = DEMO_IMG
        self.p.eval_lpips = STORE_TRUE

    def _set_exp(self, param_set="", exp_num="000"):
        self._set_default_parmas()
        self.exp_num = exp_num

        eval(f"self._set_exp_{exp_num}(param_set)")

        self.p.tag = f"{self._exp}"
        self.p.lr = self._get_lr_by_model(self.p.model_type)
        self.p.up_folder_name = self._tag

    def _convert_dict_args_list(self):
        args_dic = vars(self.p)
        args_list = []
        for key, val in args_dic.items():
            args_list.append(f"--{key}")
            if val is not STORE_TRUE:
                args_list.append(str(val))
        self._print_args_list(args_list)
        return args_list

    def export_args_list(self):
        return self._convert_dict_args_list()

    def export_cmd_str(self, use_cuda=[0]):
        args_list = self._convert_dict_args_list()
        script = "python main.py " + " ".join(args_list)
        script = self.add_cuda_visible_to_script(script, use_cuda)
        return script

    @staticmethod
    def add_cuda_visible_to_script(script, use_cuda=[0]):
        visible_devices: str = ",".join(map(str, use_cuda))
        return f"CUDA_VISIBLE_DEVICES={visible_devices} {script}"

    @staticmethod
    def _print_args_list(args_list):
        print("#" * 10 + "print for vscode debugger" + "#" * 10)
        for item in args_list:
            print(f'"{item}",')

    def _get_lr_by_model(self, model):
        if model == "gauss" or model == "wire":
            return 5e-3
        elif model == "siren":
            return 1e-4  # 1e-4 | 5e-4
        elif model == "finer":
            return 5e-4
        elif model == "pemlp":
            return 1e-3
        else:
            raise NotImplementedError

    def _use_single_data(self, pic_index="02", datasets = DIV2K_TEST):
        if hasattr(self.p, "multi_data"):
            delattr(self.p, "multi_data")

        self.p.input_path = os.path.join(datasets, f"{pic_index}.png")
        self._tag += f"_single_{pic_index}"

    def _use_datasets(self, type="div2k_test"):
        self.p.multi_data = STORE_TRUE
        if type == "div2k_test":
            self.p.input_path = DIV2K_TEST
        elif type == "div2k_train":
            self.p.input_path = DIV2K_TRAIN
        elif type == "kodak":
            self.p.input_path = KODAK
        elif type == "text_test":
            self.p.input_path = TEXT_TEST
        self._tag += f"_{type}"

    ####################################################################################
    def _set_exp_000(self, param_set):
        self._tag = "000_demo"        
        self._exp = "demo"
        self.p.transform = "sym_power"

        self.p.input_path = DEMO_IMG
        # self.p.num_epochs = 100
        # self.p.log_epoch = 50



    def _set_exp_001(self, param_set):
        self._tag = "001_trans"
        self._param_set_list = [
            "01_norm",
            "z_score",

            "gamma_0.5",
            "gamma_2.0",

            "scale_0.5",
            "scale_1.0",
            "scale_2.0",

            "inverse",
            "rpp",
            "box_cox",
            
            "sym_power",
            
        ]
        
        self._exp = param_set
       

        if param_set == "01_norm":
            self.p.transform = "min_max"
            self.p.trans_shift = 0
            self.p.trans_scale = 1
        
        elif param_set == "gamma_0.5":
            self.p.transform = "sym_power"
            self.p.gamma_trans = 0.5

        elif param_set == "gamma_2.0":
            self.p.transform = "sym_power"
            self.p.gamma_trans = 2.0
        
        elif param_set == "scale_0.5":
            self.p.transform = "min_max"
            self.p.trans_scale = 2 * 0.5
        
        elif param_set == "scale_1.0":
            self.p.transform = "min_max"
            self.p.trans_scale = 2 * 1.0
        
        elif param_set == "scale_2.0":
            self.p.transform = "min_max"
            self.p.trans_scale = 2 * 2.0
        
        elif param_set == "inverse":
            self.p.inverse = STORE_TRUE
            self.p.transform = "min_max"

        elif param_set == "rpp":
            self.p.rpp = STORE_TRUE
            self.p.transform = "min_max"
        
        else:
            self.p.transform = param_set

        self.p.input_path = DEMO_IMG
        # self.p.multi_data = STORE_TRUE


    