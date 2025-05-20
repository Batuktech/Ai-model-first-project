import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(data):
    # Metrikiig tootsooloh funkts
    # YouTube videoni undusen uzuuleltuudiig bodoh
    
    metrics = {}
    
    # Niit uzelt uil yavdlaar
    # Uil yavdal bolgonii niit uzeltiin toog tootsoolj, ih uzeltees baga ruu erembeleh
    event_total_views = data.groupby('event')['view_count'].sum().reset_index()
    event_total_views = event_total_views.sort_values('view_count', ascending=False)
    metrics['event_total_views'] = event_total_views
    
    # Video tutmiin dundaj uzelt uil yavdlaar
    # Uil yavdal bolgonii dundaj uzeltiin toog tootsoolj, ih uzeltees baga ruu erembeleh
    event_avg_views = data.groupby('event')['view_count'].mean().reset_index()
    event_avg_views = event_avg_views.sort_values('view_count', ascending=False)
    metrics['event_avg_views'] = event_avg_views
    
    # Oroltsoonii harts (laikuud + setgegdeluud huvaah uzelt)
    # Oroltsoonii harts ni laikuud ba setgegdeluudiin niilber huvaah uzeltiin too
    data['engagement_ratio'] = (data['like_count'] + data['comment_count']) / data['view_count']
    event_engagement = data.groupby('event')['engagement_ratio'].mean().reset_index()
    event_engagement = event_engagement.sort_values('engagement_ratio', ascending=False)
    metrics['event_engagement'] = event_engagement
    
    # Udur tutmiin uzelt uil yavdlaar
    # Uil yavdal bolgonii udur tutmiin dundaj uzeltiin too
    event_views_per_day = data.groupby('event')['views_per_day'].mean().reset_index()
    event_views_per_day = event_views_per_day.sort_values('views_per_day', ascending=False)
    metrics['event_views_per_day'] = event_views_per_day
    
    return metrics

def analyze_trends(data):
    """Analyze trends in YouTube video data"""
    # YouTube videoni trend shinjilgee hiih funkts
    
    trends = {}
    
    # Video heden udur huchintei bolsniig uzeltiin toond zereltseed shinjih
    # Videog niitelsen hugatsaa ba uzeltiin too hoorond yamar hamaaral baina
    age_view_corr = data['days_since_publishing'].corr(data['view_count'])
    trends['age_view_correlation'] = age_view_corr
    
    # Laikuud, setgegdeluud, uzeltiin too hoorondiin hamaaral shinjilgee
    # Laikiin too uzeltiin tootoi yamar holbootoi bolohiig todorhoiloh
    like_view_corr = data['like_count'].corr(data['view_count'])
    comment_view_corr = data['comment_count'].corr(data['view_count'])
    trends['like_view_correlation'] = like_view_corr
    trends['comment_view_correlation'] = comment_view_corr
    
    # Videoni nasnii buseer uzeltiin toog shinjilgee
    # Videog niitelsen hugatsaanii busuudeer bagtslaad, bus tutmiin uzeltiin toog shinjileh
    data['age_group'] = pd.cut(
        data['days_since_publishing'],
        bins=[0, 30, 90, 365, float('inf')],
        labels=['0-30 days', '31-90 days', '91-365 days', '366+ days']
    )
    
    # Uil yavdal bolon nasnii bus turul burt videoni dundaj uzeltiin too bolon too shirheg
    age_distribution = data.groupby(['event', 'age_group']).agg({
        'view_count': 'mean',
        'video_id': 'count'
    }).reset_index()
    
    age_distribution = age_distribution.rename(columns={'video_id': 'count'})
    trends['age_distribution'] = age_distribution
    
    # Uil yavdal tutmiin hamgiin ih uzelte suvguud
    # Uil yavdal bolon suvgaar buleglej, suvag tutmiin niit uzelt ba videoni toog tootsooloh
    top_channels = data.groupby(['event', 'channel']).agg({
        'view_count': 'sum',
        'video_id': 'count'
    }).reset_index()
    
    top_channels = top_channels.rename(columns={'video_id': 'video_count'})
    top_channels = top_channels.sort_values(['event', 'view_count'], ascending=[True, False])
    
    # Uil yavdal tutmiin hamgiin ih uzeltei neg suvgiig songoh
    # Uil yavdal burt hamgiin ih uzelte gantsaarshsan suvag oloh
    top_event_channels = top_channels.groupby('event').apply(
        lambda x: x.nlargest(1, 'view_count')
    ).reset_index(drop=True)
    
    trends['top_event_channels'] = top_event_channels
    
    return trends