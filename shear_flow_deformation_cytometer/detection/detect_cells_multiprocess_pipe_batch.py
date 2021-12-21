# -*- coding: utf-8 -*-
from shear_flow_deformation_cytometer.detection.includes.pipe_helpers import *

#
r_min = 6
batch_size = 10
write_clickpoints_file = False
write_clickpoints_masks = False
write_clickpoints_markers = False
copy_images = False
shared_memory_size = 500


from pipeline.copy_images import ProcessCopyImages
from pipeline.load_images import ProcessLoadImages, ProcessLoadImagesClickpoints, ProcessLoadImagesClickpointsAndMasks
from pipeline.detect_masks_cdb import ProcessReadMasksBatch
from pipeline.detect_masks_net import ProcessDetectMasksBatch
from pipeline.detect_masks_canny import ProcessDetectMasksBatchCanny
from pipeline.find_cells import ProcessFindCells
from pipeline.velocity import ProcessPairData
from pipeline.tank_treading import ProcessTankTreading
from pipeline.store_results import ResultCombiner



if __name__ == "__main__":
    from shear_flow_deformation_cytometer.detection import pipey
    from shear_flow_deformation_cytometer.includes.includes import getInputFile, read_args_pipeline, getInputFolder
    import sys
    import multiprocessing
    import argparse

    data_storage = JoinedDataStorage(shared_memory_size)

    # reading commandline arguments if executed from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default=None, help='specify an input file or folder')  # positional argument
    parser.add_argument('-n', '--network_weight', default="", help='provide an external the network weight file')
    parser.add_argument('-r', '--irregularity_filter', type=float, default=1.06, help='cells with larger irregularity (deviation from elliptical shape) are excluded')
    parser.add_argument('-s', '--solidity_filter', type=float, default=0.96, help='cells with smaller solidity are excluded')
    parser.add_argument('-f', '--force', type=bool, default=True, help='if True reevaluate already evaluated files')
    parser.add_argument('--rmin', type=float, default=r_min, help='cells smaller than rmin are excluded')
    args = parser.parse_args()

    file = args.file
    network_weight = args.network_weight
    irregularity_threshold = args.irregularity_filter
    solidity_threshold = args.solidity_filter
    print(f'run evaluation on {file} using {network_weight} filtering irr {irregularity_threshold} and sol {solidity_threshold}')
    clear_logs()

    print(sys.argv)
    video = getInputFolder(settings_name="detect_cells.py")
    print(video)

    pipeline = pipey.Pipeline(3)

    if 1:
        pipeline.add(get_items(args.force))

        if copy_images is True:
            pipeline.add(ProcessCopyImages(data_storage))

        # one process reads the documents
        #pipeline.add(process_load_images)
        pipeline.add(ProcessLoadImages(data_storage, batch_size=batch_size, write_clickpoints_file=write_clickpoints_file))
        pipeline.add(ProcessDetectMasksBatch(batch_size, network_weight, data_storage, None, write_clickpoints_masks=write_clickpoints_file and write_clickpoints_masks))

        #pipeline.add(ProcessReadMasksBatch(batch_size, network_weight, data_storage, None))
        #pipeline.add(ProcessDetectMasksBatchCanny(batch_size, network_weight, data_storage, None))
    else:
        pipeline.add(get_items(args.force))
        pipeline.add(ProcessLoadImagesClickpointsAndMasks(data_storage, batch_size=batch_size, type="mask"))

    # One process combines the results into a file.
    pipeline.add(ProcessFindCells(irregularity_threshold, solidity_threshold, data_storage, r_min=r_min, write_clickpoints_markers=write_clickpoints_file and write_clickpoints_markers, hollow_masks=True), 1)

    pipeline.add(ProcessPairData())

    #pipeline.add(ProcessTankTreading(data_storage), 3)

    pipeline.add(ResultCombiner(data_storage))

    pipeline.run(video)
