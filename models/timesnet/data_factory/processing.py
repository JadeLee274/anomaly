from typing import *
from data_factory.data_loader import PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader
from torch.utils.data import DataLoader


data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
}


def data_provider(
    args,
    flag: str,
) -> Tuple[Any, DataLoader]:
    Data = data_dict[args.data]
    # timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    # freq = args.freq

    data_set = Data(
        args=args,
        root_path = args.root_path,
        window_size=args.seq_len,
        flag=flag,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
