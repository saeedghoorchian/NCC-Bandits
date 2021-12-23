from datetime import datetime
import plotly.graph_objects as go


def get_average_ctr_plot(article_ctrs: dict):
    """

    """
    fig = go.Figure()

    for art_id in sorted(article_ctrs):
        ctr = article_ctrs[art_id][0]
        tss = article_ctrs[art_id][1]
        times = [datetime.fromtimestamp(ts) for ts in tss]
        fig.add_scatter(x=times, y=ctr, mode='lines', name=art_id)

    return fig