from src.utils.argument_parser import ArgumentParser


class DLIOPostProcessor:
    def __init__(self, iostat_file) -> None:
        self.arg_parser = ArgumentParser.get_instance()
        self.output_folder = self.arg_parser.output_folder
        self.comm_size = self.arg_parser.comm_size
        self.iostat_file = iostat_file
    
    def generate_report():
        # For each rank, gather stats
        
        # parse iostat report