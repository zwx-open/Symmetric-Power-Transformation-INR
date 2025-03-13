import os


def debug(use_cuda=0):
    args = [
        "--model_type",
        "siren",
        "--input_path",
        "./data/div2k/test_data/00.png",
        "--eval_lpips",
        "--transform",
        "sym_power",
        "--tag",
        "debug_demo",
        "--lr",
        "0.0001",
        "--up_folder_name",
        "000_demo",
    ]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_cuda) 
    script = "python main.py " + " ".join(args)
    print(f"Running: {script}")
    os.system(script)


if __name__ == "__main__":
    debug(0)
