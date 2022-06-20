import sys

sys.path.append("..")


import gc
import pickle

import dataset


def main():
    large_data = dataset.Dataset()
    large_data.fill_yahoo_events_second_version_r6b(
        filenames=[
            "../dataset/r6b/ydata-fp-td-clicks-v2_0.20111002",
            "../dataset/r6b/ydata-fp-td-clicks-v2_0.20111003",
            "../dataset/r6b/ydata-fp-td-clicks-v2_0.20111004",
            "../dataset/r6b/ydata-fp-td-clicks-v2_0.20111005",
            "../dataset/r6b/ydata-fp-td-clicks-v2_0.20111006",
        ],
        filtered_ids=(),
    )
    print(f"Events: {len(large_data.events)}, articles: {len(large_data.articles)}")

    with open("../dataset/r6b/subsample/data_large.pickle", "wb") as f:
        gc.disable()
        pickle.dump(large_data, f, protocol=-1)
        gc.enable()


if __name__ == "__main__":
    main()
