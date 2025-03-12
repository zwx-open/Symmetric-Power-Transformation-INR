import os

from util import misc
from util.tensorboard import writer
from util.logger import log
from util.recorder import recorder
from opt import Opt
from trainer.img_trainer import ImageTrainer
# from trainer.series_trainer import SeriesTrainer
# from trainer.img_sampling_trainer import ImageSamplingTrainer
# from trainer.sdf_trainer import SDFTrainer
# from trainer.audio_trainer import AudioTrainer
# from trainer.nerf_trainer import NeRFTrainer
import copy
import torch


def main():
    # args = opt()
    opt = Opt()
    args = opt.get_args()

    misc.fix_seed(args.seed)
    log.inst.success("start")
    writer.init_path(args.save_folder)
    log.set_export(args.save_folder)

    if args.multi_data:
        # direct input
        assert os.path.isdir(args.input_path)
        dir = args.input_path
        entries = os.listdir(dir)
        files = [
            entry for entry in entries if os.path.isfile(os.path.join(dir, entry))
        ]
        samples = sorted(files, key=lambda x: int(x.split(".")[0]))
        process_task(samples, args, cuda_num=0)

    else:
        assert os.path.isfile(args.input_path)
        start_trainer(args)

    time_dic = log.end_all_timer()
    table = recorder.add_main_table()
    print('table: ', table)

    
    if table:
        
        # _epochs = (int(args.num_epochs*0.2),int(args.num_epochs*0.4),int(args.num_epochs*1.0))
        # _attrs = ("psnr", "ssim", "lpips")
        # recorder.add_summary_table(_epochs, _attrs)
        
        recorder.dic["time"] = time_dic
        recorder.add_time_table()
        recorder.dump_table(os.path.join(args.save_folder, "res_table.md"))

    writer.close()
    log.inst.success("Done")


def process_task(sample_list, args, cuda_num=0):
    # torch.cuda.set_device(cuda_num)
    torch.set_num_threads(16)
    results = []
    for sample in sample_list:
        cur_args = copy.deepcopy(args)
        cur_args.device = f"cuda:{cuda_num}"

        if args.signal_type == "radiance_field":
            cur_args.nerf_scene = sample
        else:
            # direct input
            cur_args.input_path = os.path.join(args.input_path, sample)
        
        cur_res = start_trainer(cur_args)
        results.append(cur_res)
    return results


def start_trainer(args):
    if args.signal_type == "series":
        raise NotImplementedError
    # elif args.signal_type == "audio":
    #     trainer = AudioTrainer(args)
    elif args.signal_type == "image":
        trainer = ImageTrainer(args)
    trainer.train()
    res = getattr(trainer, "result", None)
    return res


if __name__ == "__main__":
    main()
