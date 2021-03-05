""" Utilities for loading training and game parameters, and for setting up the sql database
"""

import os
import yaml
from sqlalchemy import create_engine


class Run_Config:
    def __init__(self, config_file, **kwargs):

        with open(config_file, 'r') as f:
            #self.config_map = yaml.safe_load(f)
            # neat trick from here: https://stackoverflow.com/a/60283894/7483950
            self.config_map = yaml.safe_load(os.path.expandvars(f.read()))

        # TODO overwrite settings in the config file if they were passed in via kwargs
        # Settings for setting up scripts to run everything
        self.run_config = self.config_map.get('run_config',{})
        self.hpc_config = self.run_config.get('hpc_config')
        self.run_id = self.run_config.get('run_id','test')

        # Settings specific to the problem at hand
        self.problem_config = self.config_map.get('problem_config',{})
        # Settings for training the policy model
        self.alphazero_config = self.config_map.get('alphazero_config',{})
        self.mcts_config = self.config_map.get('mcts_config',{})

        # As a convenience, I used <config_file> as a placeholder to that own file's path.
        # Replace it here:
        self.replace_placeholder(self.run_config, '<config_file>', str(config_file))

    def replace_placeholder(self, curr_dict, placeholder, replacement):
        """ Recursively replace the placeholder string in-place in the dictionary
        """
        for key, val in curr_dict.items():
            if isinstance(val, str):
                curr_dict[key] = val.replace(placeholder, replacement)
            elif isinstance(val, dict):
                self.replace_placeholder(val, placeholder, replacement)

# def load_config_file(config_file):
#     with open(config_file, 'r') as conf:
#         config_map = yaml.load(conf)

#     return config_map

    def start_engine(self):
        self.engine = Run_Config.start_db_engine(
            **self.config_map.get('sql_database',{}))
        return self.engine

    @staticmethod
    def start_db_engine(**kwargs):
        """ Connect to the sql database that will store the game and reward data 
        used by the policy model and game runner
        """
        drivername = kwargs.get('drivername', 'sqlite')
        db_file = kwargs.get('db_file', 'game_data.db')
        if drivername == 'sqlite':
            engine = create_engine(
                f'sqlite:///{db_file}',
                # The 'check_same_thread' option only works for sqlite
                connect_args={'check_same_thread': False},
                execution_options={"isolation_level": "AUTOCOMMIT"})
        else:
            engine = Run_Config.start_server_db_engine(**kwargs)

        return engine

    @staticmethod
    def start_server_db_engine(
            drivername="postgresql+psycopg2", dbname='bde',
            port=None, host="yuma.hpc.nrel.gov", user="rlops",
            passwd_file=None, passwd=None,
            **kwargs):
        # add the ':' to separate host from port
        port = ":"+str(port) if port is not None else ""

        if passwd_file is not None:
            # read the password from a file
            with open(passwd_file, 'r') as f:
                passwd = f.read().strip()
        # add the ':' to separate user from passwd
        passwd = ":"+str(passwd) if passwd is not None else ""

        engine_str = f'{drivername}://{user}{passwd}@{host}{port}/{dbname}'
        print(f'connecting to database using: {engine_str}')
        engine = create_engine(
            engine_str, execution_options={"isolation_level": "AUTOCOMMIT"})
        return engine