import numpy as np
import fileinput
from dataclasses import dataclass
import random


@dataclass
class Event:
    """Wrapper class for one User click event

    At every timestamp 1 article is displayed to the user.
    Bandit algorithm has to choose from a pool of articles available at this timestamp.
    """
    timestamp: int
    displayed_pool_index: int  # index of article relative to the pool (which article from pool was chosen)
    user_click: int  # 1 if clicked, 0 otherwise
    user_features: list
    pool_indexes: list  # Which articles were available at (not article id, but number of arm)


class Dataset:
    """Wrapper class for Yahoo! Front Page Today Module User Click Log Dataset R6"""

    def __init__(self):
        self.articles: list[int] = []
        self.features: list[float] = []
        self.events: list[Event] = []
        self.n_arms: int = 0
        self.n_events: int = 0

    def fill_yahoo_events(self, filenames: list[str], filtered_ids: list[str], subsample_percentage: float = 1.0):
        """
        Reads a stream of events from the list of given files.

        Args:
            filenames: List of filenames.
            filtered_ids: Article ids to be filtered out of the data.
            subsample_percentage: which portion of data to subsample.
        """

        assert 0.0 <= subsample_percentage <= 1.0, f"Subsample percentage is {subsample_percentage}" \
                                                   f"should be in [0.0; 1.0] range.'"

        self.articles = []
        self.features = []
        self.events = []

        skipped = 0
        skipped_no_articles = 0

        with fileinput.input(files=filenames) as f:
            for line in f:
                cols = line.split()
                if (len(cols) - 10) % 7 != 0:
                    # Some log files contain rows with erroneous data.
                    skipped += 1
                    continue

                if random.random() > subsample_percentage:
                    continue

                pool_idx = []
                pool_ids = []

                timestamp = int(cols[0])

                displayed_article_id = cols[1]
                if displayed_article_id in filtered_ids:
                    continue

                user_click = int(cols[2])  # 1 for click, 0 for no click

                # Next 7 columns are "|user <user_feature_1> ... <user_feature_6>"
                # Each <user_feature_i> looks like this "i:0.000012"
                user_features = [float(x[2:]) for x in cols[4:10]]

                # After the first 10 columns are the articles and their features.
                # Each article has 7 columns (article id preceeded by | and 6 features.
                for i in range(10, len(cols) - 6, 7):
                    # First symbol is "|"
                    id = cols[i][1:]
                    if id in filtered_ids:
                        continue
                    if id not in self.articles:
                        self.articles.append(id)
                        self.features.append([float(x[2:]) for x in cols[i + 1: i + 7]])
                    pool_idx.append(self.articles.index(id))
                    pool_ids.append(id)

                if len(pool_idx) <= 1:
                    # print("\n\n\nWARNING!\n\n\nYour strict filtering led to some"
                          # f"events having not enough articles to choose from, event number {len(self.events)}")
                    skipped_no_articles += 1
                    continue

                self.events.append(
                    Event(
                        timestamp=timestamp,
                        displayed_pool_index=pool_ids.index(displayed_article_id),
                        user_click=user_click,
                        user_features=user_features,
                        pool_indexes=pool_idx,
                    )
                )
        self.features = np.array(self.features)
        self.n_arms = len(self.articles)
        self.n_events = len(self.events)
        print(self.n_events, "events with", self.n_arms, "articles, from files ", filenames)
        if skipped:
            print(f"Skipped events: {skipped}"), skipped
        if skipped_no_articles:
            print(f"Skipped events because of no arms to choose from {skipped_no_articles}")

    def get_article_average_ctr(self, article_index: int, average_over: int) -> tuple:
        """Get average click through rate for one article in the dataset.

        Returns:
            tuple(list of ctrs, list of timestamps): average click through rates and corresponding timestamps.
        """
        art_events = [
            ev for ev in self.events
            if ev.pool_indexes[ev.displayed_pool_index] == article_index
        ]

        num = 0
        average_ctrs = []
        average_timestamps = []
        ctr = 0
        timestamps = []
        for i, event in enumerate(art_events):
            num += 1
            ctr += event.user_click
            timestamps.append(event.timestamp)
            if num == average_over:
                num = 0
                average_ctrs.append(ctr / average_over)
                average_timestamps.append(np.mean(timestamps))
                ctr = 0
                timestamps = []
        # Last measurements are ignored if there are less than 4000 of them

        return average_ctrs, average_timestamps

    def gather_all_average_ctrs(self, average_over: int) -> dict:
        """For all the articles in the dataset gather their average ctr in a dict.

        Returns:
            dict: {
                <article_id>: (list ctrs, list timestamps)
            }
        """
        article_indices = range(len(self.articles))
        article_ctrs = {}
        for art_index in article_indices:
            art_id = self.articles[art_index]
            ctrs, tss = self.get_article_average_ctr(art_index, average_over)
            article_ctrs[art_id] = (ctrs, tss)
        return article_ctrs

    def __repr__(self):
        return f"Dataset(n_arms={self.n_arms},n_events={self.n_events})"
