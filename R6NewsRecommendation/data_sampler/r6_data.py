import logging
import os

import numpy as np
import fileinput

from tqdm import tqdm
from random import choices, shuffle

from util.base_config import BaseConfig


class R6DataSamplerConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "raw_dir": str,
            "processed_dir": str,
            "num_contexts": int,
            "day": list,
            "reward_balance": float,
        }


class R6DataSampler:
    def __init__(self, config, logger):
        self.raw_dir = config.raw_dir
        self.processed_dir = config.processed_dir
        self.num_contexts = config.num_contexts
        self.day = config.day
        self.reward_balance = config.reward_balance
        self.num_contexts = config.num_contexts
        self.logger = logger

    def sample(self):
        if self.reward_balance < 0:
            events, features, num_arms = get_data(
                data_dir=self.raw_dir,
                save_dir=self.processed_dir,
                day=self.day,
                logger=self.logger,
                num_contexts=self.num_contexts,
                split_zero_one=False
            )
        else:
            events_0, events_1, features, num_arms = get_data(
                data_dir=self.raw_dir,
                save_dir=self.processed_dir,
                day=self.day,
                logger=self.logger,
                num_contexts=self.num_contexts,
                split_zero_one=True
            )
            events = merge_events(events_0, events_1, self.reward_balance)

        return events, features, num_arms


"""
Line format for yahoo events:
1241160900 109513 0 |user 2:0.000012 3:0.000000 4:0.000006 5:0.000023 6:0.999958 1:1.000000 |109498 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 |109509 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 [[...more article features omitted...]] |109453 2:0.421669 3:0.000011 4:0.010902 5:0.309585 6:0.257833 1:1.000000

Some log files contain rows with erroneous data.

After the first 10 columns are the articles and their features.
Each article has 7 columns (articleid + 6 features)
Therefore number_of_columns-10 % 7 = 0
"""
def get_yahoo_events(filenames, logger, num_contexts=-1):
    """
    Reads a stream of events from the list of given files.
    
    Parameters
    ----------
    filenames : list
        List of filenames
    
    Stores
    -------    
    articles : [article_ids]
    features : [[article_1_features] .. [article_n_features]]
    events : [
                 0 : displayed_article_index (relative to the pool),
                 1 : user_click,
                 2 : [user_features],
                 3 : [pool_indexes]
             ]
    """
    day = [filename[-1] for filename in filenames]
    logger.info("Get Yahoo Events Dataset - Day " + ", ".join(day))

    articles = []
    features = []
    events = []

    skipped = 0

    # with fileinput.input(files=filenames) as f:
    for filename in filenames:
        with open(filename, "r") as f:
            for idx, line in enumerate(tqdm(f)):
                if (num_contexts > 0) and (idx >= num_contexts):
                    break
                cols = line.split()
                if (len(cols) - 10) % 7 != 0:
                    skipped += 1
                else:
                    pool_idx = []
                    pool_ids = []
                    for i in range(10, len(cols) - 6, 7):
                        id = cols[i][1:]
                        if id not in articles:
                            articles.append(id)
                            features.append([float(x[2:]) for x in cols[i + 1: i + 7]])
                        pool_idx.append(articles.index(id))
                        pool_ids.append(id)

                    events.append(
                        [
                            pool_ids.index(cols[1]),
                            int(cols[2]),
                            [float(x[2:]) for x in cols[4:10]],
                            pool_idx,
                        ]
                    )
    features = np.array(features)
    num_arms = len(articles)
    n_events = len(events)
    logger.info(f"{n_events} events with {num_arms} articles")
    if skipped != 0:
        logger.info(f"Skipped events: {skipped}")

    return articles, features, events, num_arms, n_events


def save_yahoo_events_nonsplit(data_dir, save_dir, day, logger, num_contexts):
    logger.info(f"Start Preprocessing Yahoo Events Dataset - Day " + ", ".join(day))
    filename = [f"{data_dir}/ydata-fp-td-clicks-v1_0.2009050" + str(d) for d in day]
    save_file = "D" + "".join(day) + ".npy"
    if os.path.isfile(os.path.join(save_dir, save_file)):
        logger.info("Dataset Already Preprocessed - Day " + ", ".join(day))
        logger.info(f"Saved at {os.path.join(save_dir, save_file)}")
    else:
        os.makedirs(save_dir, exist_ok=True)
        articles, features, events, num_arms, n_events = get_yahoo_events(filename, logger, num_contexts)
        np.save(file=os.path.join(save_dir, save_file), arr=(articles, features, events, num_arms, n_events), allow_pickle=True)
        logger.info(f"Processed Yahoo Events Dataset - Day " + ", ".join(day))
    logger.info(f"Saved at {os.path.join(save_dir, save_file)}")


def save_yahoo_events_split(data_dir, save_dir, day, logger, num_contexts):
    logger.info(f"Start Preprocessing & Splitting Yahoo Events Dataset - Day " + ", ".join(day))
    filename = [f"{data_dir}/ydata-fp-td-clicks-v1_0.2009050" + str(d) for d in day]
    save_file_0 = "D" + "".join(day) + "-R0.npy"
    save_file_1 = "D" + "".join(day) + "-R1.npy"
    if (os.path.isfile(os.path.join(save_dir, save_file_0)) and os.path.isfile(os.path.join(save_dir, save_file_1))):
        logger.info("Dataset Already Preprocessed - Day " + ", ".join(day))
    else:
        os.makedirs(save_dir, exist_ok=True)
        articles, features, events, num_arms, n_events = get_yahoo_events(filename, logger, num_contexts)
        events_0 = []
        events_1 = []
        for event in events:
            reward = event[1]
            if reward == 0:
                events_0.append(event)
            else:
                events_1.append(event)
        n_events_0 = len(events_0)
        n_events_1 = len(events_1)
        np.save(file=os.path.join(save_dir, save_file_0), arr=(articles, features, events_0, num_arms, n_events_0),
                allow_pickle=True)
        np.save(file=os.path.join(save_dir, save_file_1), arr=(articles, features, events_1, num_arms, n_events_1),
                allow_pickle=True)
        logger.info(f"Processed & Split Yahoo Events Dataset - Day " + ", ".join(day))
    logger.info(f"Data with Reward 0 Saved at {os.path.join(save_dir, save_file_0)}")
    logger.info(f"Data with Reward 1 Saved at {os.path.join(save_dir, save_file_1)}")


def get_data(data_dir, save_dir, day, logger, num_contexts, split_zero_one=False):
    """
    dataset
    """
    if not split_zero_one:
        return get_data_nonsplit(data_dir, save_dir, day, logger, num_contexts)
    else:
        return get_data_split(data_dir, save_dir, day, logger, num_contexts)


def get_data_nonsplit(data_dir, save_dir, day, logger, num_contexts):
    day = list(map(str, day))
    file = "D" + "".join(day) + ".npy"
    logger.info("Loading Yahoo Events Dataset - Day " + ", ".join(day))
    if not os.path.exists(os.path.join(save_dir, file)):
        logger.info("No dataset found. Start Preprocessing Yahoo Events Dataset - Day " + ", ".join(day))
        save_yahoo_events_nonsplit(data_dir=data_dir, save_dir=save_dir, day=day, logger=logger, num_contexts=num_contexts)
    articles, features, events, num_arms, n_events = np.load(os.path.join(save_dir, file), allow_pickle=True)
    logger.info("Loading Finished: " + f"{n_events} events with {num_arms} articles")
    logger.info("")

    return events, features, num_arms


def get_data_split(data_dir, save_dir, day, logger, num_contexts):
    day = list(map(str, day))
    file_0 = "D" + "".join(day) + "-R0.npy"
    file_1 = "D" + "".join(day) + "-R1.npy"
    logger.info("Loading Split Yahoo Events Dataset - Day " + ", ".join(day))
    if not (os.path.exists(os.path.join(save_dir, file_0)) and os.path.exists(os.path.join(save_dir, file_1))):
        logger.info("No balance dataset found. Start Preprocessing & Splitting Yahoo Events Dataset - Day " + ", ".join(day))
        save_yahoo_events_split(data_dir=data_dir, save_dir=save_dir, day=day, logger=logger, num_contexts=num_contexts)
    articles, features, events_0, num_arms, n_events_0 = np.load(os.path.join(save_dir, file_0), allow_pickle=True)
    articles, features, events_1, num_arms, n_events_1 = np.load(os.path.join(save_dir, file_1), allow_pickle=True)
    logger.info("Loading Finished: " + f"{n_events_0} R0 events, {n_events_1} R1 events with {num_arms} articles")
    logger.info("")

    return events_0, events_1, features, num_arms


def merge_events(events_0, events_1, ratio=0.5):
    total_len = len(events_0) + len(events_1)
    len_1 = int(total_len * ratio)
    len_0 = total_len - len_1
    events = choices(population=events_0, k=len_0) + \
        choices(population=events_1, k=len_1)
    shuffle(events)
    return events



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--day", nargs='+', default=[3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--root", type=str, default="/data/R6/")
    args = parser.parse_args()
    save_yahoo_events_nonsplit(save_dir=args.root, day=args.day, logger=logging.getLogger(__name__))
