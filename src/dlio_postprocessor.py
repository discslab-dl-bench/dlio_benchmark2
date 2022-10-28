import os
import re
import json
from ssl import get_server_certificate
import pandas as pd
from statistics import mean, median, stdev, quantiles
from src.utils.argument_parser import str2bool

import argparse

class DLIOPostProcessor:
    def __init__(self, args) -> None:
        self.args = args
        self.outdir = self.args.output_folder
        self.comm_size = self.args.num_proc
        self.epochs = self.args.epochs
        self.epochs_list = [str(e) for e in range(1, self.epochs + 1)]

        self.do_eval = self.args.do_eval
        self.do_checkpoint = self.args.checkpoint

        self.batch_size = self.args.batch_size
        self.batch_size_eval = self.args.batch_size_eval

        self.verify_and_load_all_files()

        # Overall report statistics
        self.overall_samples_per_second = 0
        self.epoch_samples_per_second = {}

        self.r_bandwidth_per_epoch = {}
        self.w_bandwidth_per_epoch = {}
        self.r_overall_bandwidth = []
        self.w_overall_bandwidth = []

        # I think it might be better to fill these two dicts up as we go along
        # and print them out at the end. That would follow the expected structure of the report
        self.overall_stats = {}
        # self.per_epoch_stats = {}

        self.epochs_with_evals = set()
        self.epochs_with_ckpts = set()


    def verify_and_load_all_files(self):
        outdir_listing = [f for f in os.listdir(self.outdir) if os.path.isfile(os.path.join(self.outdir, f))]

        all_files = ['iostat.json', 'per_epoch_stats.json']

        load_and_proc_time_files = []

        for rank in range(self.comm_size):
            load_and_proc_time_files.append(f'{rank}_load_and_proc_times.json')

        all_files.extend(load_and_proc_time_files)

        is_missing_file = False
        for necessary_file in all_files:
            if necessary_file not in outdir_listing:
                print(f"ERROR: missing necessary file: {os.path.join(self.outdir, necessary_file)}")
        if is_missing_file:
            exit(-1)

        # All files are present
        self.iotrace = json.load(open(os.path.join(self.outdir, 'iostat.json'), 'r'))
        self.per_epoch_stats = json.load(open(os.path.join(self.outdir, 'per_epoch_stats.json'), 'r'))
        # These ones will be loaded in later
        self.load_and_proc_time_files = [os.path.join(self.outdir, f) for f in load_and_proc_time_files]


    def process_loading_and_processing_times(self):
        
        self.epoch_loading_times = {}
        self.epoch_processing_times = {}
        all_loading_times = []
        all_processing_times = []

        # There is one file per worker process, with data
        # separated by epoch and by phase of training (block, eval)
        # First, we will combine the different workers' data before
        # computing overall and per training phase statistics.
        for file in self.load_and_proc_time_files:
            load_and_proc_times = json.load(open(file, 'r'))
            
            for epoch in self.epochs_list:

                loading_data = load_and_proc_times[epoch]['load']

                # Initialize a dictionary to hold the data if not present
                if epoch not in self.epoch_loading_times:
                    self.epoch_loading_times[epoch] = {}

                # For each training phase, fetch the loading times and combine them
                for phase, phase_loading_times in loading_data.items():
                    assert isinstance(phase_loading_times, list)

                    if re.match(r'eval', phase):
                        batch_size = self.batch_size_eval
                    else:
                        batch_size = self.batch_size
                    
                    phase_loading_times = phase_loading_times / batch_size
                    all_loading_times.extend(phase_loading_times)

                    if phase not in self.epoch_loading_times[epoch]:
                        self.epoch_loading_times[epoch][phase] = phase_loading_times
                    else:
                        self.epoch_loading_times[epoch][phase].extend(phase_loading_times)

                # Same thing for processing times
                processing_data = load_and_proc_times[epoch]['proc']

                # Initialize a dictionary to hold the data if not present
                if epoch not in self.epoch_processing_times:
                    self.epoch_processing_times[epoch] = {}

                # For each training phase, fetch the loading times and combine them
                for phase, phase_procesing_times in processing_data.items():
                    assert isinstance(phase_procesing_times, list)

                    all_processing_times.extend(phase_procesing_times)

                    if phase not in self.epoch_processing_times[epoch]:
                        self.epoch_processing_times[epoch][phase] = phase_procesing_times
                    else:
                        self.epoch_processing_times[epoch][phase].extend(phase_procesing_times)


        # At this point, we should have one big structure containing overall stats, 
        # as well as all the combined loading and processing times for each phase of training
        
        # Save the overall stats
        self.overall_stats['samples/s'] = self.get_stats(all_loading_times)
        self.overall_stats['sample_latency'] = self.get_stats(all_processing_times)
        self.overall_stats['avg_loading_time'] = '{:.2f}'.format(sum(all_loading_times) / self.comm_size)
        self.overall_stats['avg_processing_time'] = '{:.2f}'.format(sum(all_processing_times) / self.comm_size)

        # Save the stats for each phase of training
        for epoch in self.epochs_list:
            epoch_loading_data = self.epoch_loading_times[epoch]
            epoch_processing_data = self.epoch_processing_times[epoch]

            for phase, loading_times in epoch_loading_data.items():
                self.per_epoch_stats[epoch][phase]['samples/s'] = self.get_stats(loading_times)
                self.per_epoch_stats[epoch][phase]['avg_loading_time'] = '{:.2f}'.format(sum(loading_times) / self.comm_size)

            for phase, processing_times in epoch_processing_data.items():
                self.per_epoch_stats[epoch][phase]['sample_latency'] = self.get_stats(processing_times)
                self.per_epoch_stats[epoch][phase]['avg_processing_time'] = '{:.2f}'.format(sum(processing_times) / self.comm_size)
  

    def get_epoch_time_ranges(self):
        self.epoch_timeranges = {}
        self.eval_timeranges = {}
        self.ckpt_timeranges = {}

        epoch_start_end_ts = pd.read_csv(os.path.join(self.outdir, 'epoch_time_ranges.csv'), names = ['epoch_num', 'start', 'end'])
        epoch_start_end_ts.start = pd.to_datetime(epoch_start_end_ts.start)
        epoch_start_end_ts.end = pd.to_datetime(epoch_start_end_ts.end)
        if self.do_eval:
            eval_start_end_ts = pd.read_csv(os.path.join(self.outdir, 'eval_time_ranges.csv'), names = ['epoch_num', 'start', 'end'])
            eval_start_end_ts.start = pd.to_datetime(eval_start_end_ts.start)
            eval_start_end_ts.end = pd.to_datetime(eval_start_end_ts.end)
            self.epochs_with_evals = set(eval_start_end_ts['epoch_num'])
        if self.do_checkpoint:
            ckpt_start_end_ts = pd.read_csv(os.path.join(self.outdir, 'ckpt_time_ranges.csv'), names = ['epoch_num', 'start', 'end'])
            ckpt_start_end_ts.start = pd.to_datetime(ckpt_start_end_ts.start)
            ckpt_start_end_ts.end = pd.to_datetime(ckpt_start_end_ts.end)
            self.epochs_with_ckpts = set(ckpt_start_end_ts['epoch_num'])

        # For each epoch, extract the start and end times and store them in a dictionary
        for epoch in range(1, self.epochs+1):
            timerange = epoch_start_end_ts[epoch_start_end_ts['epoch_num'] == epoch]
            # Extract the values from the returned pd.Series objects
            self.epoch_timeranges[epoch] = (timerange['start'].iloc[0], timerange['end'].iloc[0])

            duration = timerange['end'].iloc[0] - timerange['start'].iloc[0]
            self.per_epoch_stats[epoch] = {
                'start': timerange['start'].iloc[0],
                'end': timerange['end'].iloc[0],
                'duration': duration.total_seconds()
            }
            # Save additional values for evals and checkpoiting, if relevant
            if self.do_eval and epoch in self.epochs_with_evals:
                timerange = eval_start_end_ts[eval_start_end_ts['epoch_num'] == epoch]
                self.eval_timeranges[epoch] = (timerange['start'].iloc[0], timerange['end'].iloc[0])
                duration = timerange['end'].iloc[0] - timerange['start'].iloc[0]
                self.per_epoch_stats[epoch]['eval'] = {
                    'start': timerange['start'].iloc[0],
                    'end': timerange['end'].iloc[0],
                    'duration': duration.total_seconds()
                }
            if self.do_checkpoint and epoch in self.epochs_with_ckpts:
                timerange = ckpt_start_end_ts[ckpt_start_end_ts['epoch_num'] == epoch]
                self.ckpt_timeranges[epoch] = (timerange['start'].iloc[0], timerange['end'].iloc[0])
                duration = timerange['end'].iloc[0] - timerange['start'].iloc[0]
                self.per_epoch_stats[epoch]['ckpt'] = {
                    'start': timerange['start'].iloc[0],
                    'end': timerange['end'].iloc[0],
                    'duration': duration.total_seconds()
                }
        
        # Get overall start, end and duration
        self.overall_stats['start'] = epoch_start_end_ts['start'].iloc[0] 
        self.overall_stats['end'] = epoch_start_end_ts['end'].iloc[-1] 
        self.overall_stats['duration'] = self.overall_stats['end'] - self.overall_stats['start'] 
        

    def get_stats(self, series):
        """
        Return a dictionary with various statistics of the given series
        """
        # Returns 99 cut points
        # We can use inclusive because we have the entire population
        percentiles = quantiles(series, n=100, method='inclusive')
        return {
            "mean": '{:.2f}'.format(mean(series)),
            "std": '{:.2f}'.format(stdev(series)),
            "min": '{:.2f}'.format(min(series)),
            "median": '{:.2f}'.format(median(series)),
            "p90": '{:.2f}'.format(percentiles[89]),
            "p99": '{:.2f}'.format(percentiles[98]),
            "max": '{:.2f}'.format(max(series))
        }


    def process_loading_times(self):

        epoch_loading_times = {}
        eval_loading_times = {}
        all_loading_times = []

        for file in self.loading_time_files:
            if re.match(r'.*eval.*', file):
                saveto = eval_loading_times
                effective_batch_size = self.batch_size_eval
            else:
                saveto = epoch_loading_times
                effective_batch_size = self.batch_size

            df = pd.read_csv(os.path.join(self.outdir, file), names = ['timestamp', 'epoch_num', 'loading_time'])
            # Calculate samples per second by integer dividing batch size by loading time
            df['loading_time'] = df['loading_time'].apply(lambda x: effective_batch_size / x)
            # Group by epoch number, convert to dict
            loading_times = df.groupby('epoch_num')['loading_time'].apply(list).to_dict()
            all_loading_times += df['loading_time'].to_list()
            # Merge with overall (for each epoch, we have one file per worker)
            for epoch in loading_times.keys():
                if epoch in saveto:
                    saveto[epoch].extend(loading_times[epoch])
                else:
                    saveto[epoch] = loading_times[epoch]
        
        # Save the overall stats
        self.overall_stats['samples/s'] = self.get_stats(all_loading_times)
        self.overall_stats['loading_time'] = '{:.2f}'.format(sum(all_loading_times) / self.comm_size)
        
        # Save the per epoch/eval stats
        for epoch, loading_times in epoch_loading_times.items():
            self.per_epoch_stats[epoch]['samples/s'] = self.get_stats(loading_times)
            self.per_epoch_stats[epoch]['loading_time'] = '{:.2f}'.format(sum(loading_times) / self.comm_size)

        for epoch, loading_times in eval_loading_times.items():
            self.per_epoch_stats[epoch]['eval']['samples/s'] = self.get_stats(loading_times)
            self.per_epoch_stats[epoch]['eval']['loading_time'] = '{:.2f}'.format(sum(loading_times) / self.comm_size)


    def parse_iostat_trace(self):
        """
        Parse the iostat JSON file and return disk and cpu usage information
        """
        # TODO: Support tracing on multiple hosts, here we only get data for the first
        iotrace = self.iotrace['sysstat']['hosts'][0]['statistics']
        # We will convert the iostat JSON output into a Dataframe indexed by timestamp 
        # Timestamps are already in UTC (when generated from within the container)
        # Pandas can read the format, then we can convert to numpy datetime64
        cpu_stats = pd.DataFrame(columns=['timestamp', 'user', 'system', 'iowait', 'steal', 'idle'])
        # The following columns are available:
        # ['timestamp', 'disk', 'r/s', 'w/s', 'rMB/s', 'wMB/s', 'r_await', 'w_await', 'rareq-sz', 'wareq-sz', 'aqu-sz'])
        disk_stats = pd.DataFrame(columns=['timestamp', 'disk', 'r/s', 'w/s', 'rMB/s', 'wMB/s', 'r_await', 'w_await', 'aqu-sz'])
        cpu_i = 0
        disk_i = 0
        for item in iotrace:
            ts = pd.to_datetime(item['timestamp']) + pd.DateOffset(hours=4)
            # Need to convert to UTC, this will depend on your timezone

            cpu = item['avg-cpu']
            # Combine user and nice cpu time into one for conciseness
            cpu_stats.loc[cpu_i] = [ts, cpu['user'] + cpu['nice'], cpu['system'], cpu['iowait'], cpu['steal'], cpu['idle']]
            cpu_i += 1
            # Add one row per disk
            for disk in item['disk']:
                row = [ts, disk['disk_device'], disk['r/s'], disk['w/s'], disk['rMB/s'], disk['wMB/s'], disk['r_await'], disk['w_await'], disk['aqu-sz']]
                disk_stats.loc[disk_i] = row
                disk_i += 1
        # Convert timestamp fields to datatime
        cpu_stats.timestamp = pd.to_datetime(cpu_stats.timestamp)
        disk_stats.timestamp = pd.to_datetime(disk_stats.timestamp)
        self.disk_stats = disk_stats
        self.cpu_stats = cpu_stats


    def extract_stats_from_iostat_trace(self):
        r_overall_bandwidth = []
        w_overall_bandwidth = []
        r_overall_iops = []
        w_overall_iops = []
        r_overall_wait = []
        w_overall_wait = []
        overall_aqu_sz = []

        cpu_overall_user = []
        cpu_overall_sys = []
        cpu_overall_iowait = []
        cpu_overall_steal = []
        cpu_overall_idle = []

        disk_stats_to_extract = ['rMB/s', 'wMB/s', 'r/s', 'w/s', 'r_await', 'w_await', 'aqu-sz']
        disk_accumulators = [r_overall_bandwidth, w_overall_bandwidth, r_overall_iops, w_overall_iops, r_overall_wait, w_overall_wait, overall_aqu_sz]
        cpu_stats_to_extract = ['user', 'system', 'iowait', 'steal', 'idle']
        cpu_accumulators = [cpu_overall_user, cpu_overall_sys, cpu_overall_iowait, cpu_overall_steal, cpu_overall_idle]

        def addto_and_return_stats(addto, df, stat):
            data = df[stat].to_list()
            addto += data
            return self.get_stats(data)

        for epoch in self.epochs_list:
            epoch_data = self.per_epoch_stats[epoch]

            for phase, phase_data in epoch_data.items():
                if not isinstance(phase_data, dict):
                    continue

                start, end = pd.to_datetime(phase_data['start']), pd.to_datetime(phase_data['end'])

                disk_io = self.get_series_daterange(self.disk_stats, start, end)

                for i, stat in enumerate(disk_stats_to_extract):
                    self.per_epoch_stats[epoch][phase][stat] = addto_and_return_stats(disk_accumulators[i], disk_io, stat)

                cpu_data = self.get_series_daterange(self.cpu_stats, start, end)

                self.per_epoch_stats[epoch][phase]['cpu'] = {}
                for i, stat in enumerate(cpu_stats_to_extract):
                    self.per_epoch_stats[epoch][phase]['cpu'][stat] = addto_and_return_stats(cpu_accumulators[i], cpu_data, stat)


        # Compute overall stats
        self.overall_stats['rMB/s'] = self.get_stats(r_overall_bandwidth)
        self.overall_stats['wMB/s'] = self.get_stats(w_overall_bandwidth)
        self.overall_stats['r/s'] = self.get_stats(r_overall_iops)
        self.overall_stats['w/s'] = self.get_stats(w_overall_iops)
        self.overall_stats['r_await'] = self.get_stats(r_overall_wait)
        self.overall_stats['w_await'] = self.get_stats(w_overall_wait)
        self.overall_stats['aqu-sz'] = self.get_stats(overall_aqu_sz)

        self.overall_stats['cpu'] = {
            'user': self.get_stats(cpu_overall_user),
            'system': self.get_stats(cpu_overall_sys),
            'iowait': self.get_stats(cpu_overall_iowait),
            'steal': self.get_stats(cpu_overall_steal),
            'idle': self.get_stats(cpu_overall_idle)
        }

    def get_series_daterange(self, series, start, end): 
        data = series[series['timestamp'] >= start]
        data = data[data['timestamp'] < end]
        return data


    def print_report(self):

        def print_stats(stats):
            if isinstance(stats, dict):
                stats = "\t".join(stats.values())
            return stats
                
        def print_out_section(outfile, stats_dict, has_loading=True, extra_indent=0):
            extra_tabs = '\t' * extra_indent
            outfile.write(f"\t{extra_tabs}Started:\t\t\t{stats_dict['start']}\n")
            outfile.write(f"\t{extra_tabs}Ended:\t\t\t\t{stats_dict['end']}\n")
            outfile.write(f"\t{extra_tabs}Duration (s):\t\t\t{stats_dict['duration']}\n")
            if has_loading:
                outfile.write(f"\t{extra_tabs}Avg Loading Time / Rank (s):\t{stats_dict['avg_loading_time']}\n")
                outfile.write(f"\t{extra_tabs}Avg Compute Time / Rank (s):\t{stats_dict['avg_processing_time']}\n\n")
            outfile.write(f"\t{extra_tabs}\t\t\t\tmean\tstd\tmin\tmedian\tp90\tp99\tmax\n")
            outfile.write(f"\t{extra_tabs}\t\t\t\t-------------------------------------------------------\n")
            if has_loading:
                outfile.write(f"\t{extra_tabs}Samples/s:\t\t\t{print_stats(stats_dict['samples/s'])}\n")
                outfile.write(f"\t{extra_tabs}Sample Latency (s):\t\t{print_stats(stats_dict['sample_latency'])}\n")
            outfile.write(f"\t{extra_tabs}Read Bandwidth (MB/s):\t\t{print_stats(stats_dict['rMB/s'])}\n")
            outfile.write(f"\t{extra_tabs}Write Bandwidth (MB/s):\t\t{print_stats(stats_dict['wMB/s'])}\n")
            outfile.write(f"\t{extra_tabs}Read IOPS:\t\t\t{print_stats(stats_dict['r/s'])}\n")
            outfile.write(f"\t{extra_tabs}Write IOPS:\t\t\t{print_stats(stats_dict['w/s'])}\n")
            outfile.write(f"\t{extra_tabs}Avg Read Time (ms):\t\t{print_stats(stats_dict['r_await'])}\n")
            outfile.write(f"\t{extra_tabs}Avg Write Time (ms):\t\t{print_stats(stats_dict['w_await'])}\n")
            outfile.write(f"\t{extra_tabs}Avg Queue Length:\t\t{print_stats(stats_dict['aqu-sz'])}\n")
            outfile.write(f"\t{extra_tabs}CPU usage:\n")
            outfile.write(f"\t\t{extra_tabs}User (%):\t\t{print_stats(stats_dict['cpu']['user'])}\n")
            outfile.write(f"\t\t{extra_tabs}System (%):\t\t{print_stats(stats_dict['cpu']['system'])}\n")
            outfile.write(f"\t\t{extra_tabs}IO wait (%):\t\t{print_stats(stats_dict['cpu']['iowait'])}\n")
            outfile.write(f"\t\t{extra_tabs}Steal (%):\t\t{print_stats(stats_dict['cpu']['steal'])}\n")
            outfile.write(f"\t\t{extra_tabs}Idle (%):\t\t{print_stats(stats_dict['cpu']['idle'])}\n\n")

        # Get overall start, end and duration
        self.overall_stats['start'] = self.per_epoch_stats["1"]['start']
        self.overall_stats['end'] = self.per_epoch_stats[str(self.epochs)]['end']
        duration = pd.to_datetime(self.overall_stats['end']) - pd.to_datetime(self.overall_stats['start']) 
        self.overall_stats['duration'] = '{:.2f}'.format(duration.total_seconds())

        with open(os.path.join(self.outdir, "DLIO_report.txt"), 'w') as outfile:
            outfile.write("Overall\n")
            print_out_section(outfile, self.overall_stats)

            outfile.write("Detailed Report\n")

            i_eval = i_ckpt = 1
            for epoch in self.epochs_list:
                i_blk = 1

                epoch_data = self.per_epoch_stats[epoch]
                
                outfile.write(f"Epoch {epoch}\n")

                outfile.write(f"\tStarted:\t\t{epoch_data['start']}\n")
                outfile.write(f"\tEnded:\t\t\t{epoch_data['end']}\n")
                outfile.write(f"\tDuration (s):\t\t{epoch_data['duration']}\n\n")

                for phase, phase_data in epoch_data.items():

                    if not isinstance(phase_data, dict):
                        continue
                    
                    has_loading = True

                    if re.match(r'block\d+', phase):
                        outfile.write(f"\tBlock {i_blk}\n")
                        i_blk += 1
                    elif re.match(r'eval', phase):
                        outfile.write(f"\tEval {i_eval}\n")
                        i_eval += 1
                    elif re.match(r'ckpt\d+', phase):
                        outfile.write(f"\tCheckpoint {i_ckpt}\n")
                        has_loading = False
                        i_ckpt += 1
                    else:
                        print("Warning: unknown training phase")
                        outfile.write(f"\t{phase}\n")

                    print_out_section(outfile, phase_data, has_loading=has_loading, extra_indent=1)



    def generate_report(self):
        self.process_loading_and_processing_times()
        # parse iostat report
        self.parse_iostat_trace()
        self.extract_stats_from_iostat_trace()
        # Write the report
        self.print_report()


def main():
    """
    The main method to start the benchmark runtime.
    """
    parser = argparse.ArgumentParser(description='DLIO PostProcessor')
    
    parser.add_argument("-of", "--output-folder", default="./output", type=str,
                        help="Folder containing the output of a benchmark run.")
    parser.add_argument("-np", "--num-proc", default=1, type=int,
                        help="Number of processes that were ran.")
    parser.add_argument("-e", "--epochs", default=1, type=int,
                        help="Number of epochs to be emulated within benchmark.")
    parser.add_argument("-bs", "--batch-size", default=1, type=int,
                        help="Per worker batch size for training records.")
    parser.add_argument("-de", "--do-eval", default=False, type=str2bool,
                        help="If evaluations were simulated.")
    parser.add_argument("-bse", "--batch-size-eval", default=1, type=int,
                        help="Per worker batch size for evaluation records.")
    parser.add_argument("-c", "--checkpoint", default=False, type=str2bool,
                        help="If checkpointing was simulated")

    args = parser.parse_args()
    postproc = DLIOPostProcessor(args)
    postproc.generate_report()

if __name__ == '__main__':
    main()
    exit(0)