import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime, timezone
import time
import os

def setup_youtube_api(api_key):
    """Initialize the YouTube API client"""
    # YouTube API klientiin initialize hiih
    # API tulshuurtei kholbogdoj holboogoo uusgeh
    youtube = build('youtube', 'v3', developerKey=api_key)
    return youtube

def search_event_videos(youtube, event_name, max_results=50):
    """Search for videos related to a specific event"""
    # Todorhoilson uil yavdaltai holbootoi videog haih
    # YouTube-ees todorhoilson uil yavdliig haij oloh
    request = youtube.search().list(
        q=event_name,
        part='snippet',
        type='video',
        maxResults=max_results,
        order='viewCount'
    )
    response = request.execute()
    
    videos = []
    for item in response['items']:
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        channel = item['snippet']['channelTitle']
        published_at = item['snippet']['publishedAt']
        
        videos.append({
            'video_id': video_id,
            'title': title,
            'channel': channel,
            'published_at': published_at,
            'event': event_name
        })
    
    return videos

def get_video_statistics(youtube, video_ids):
    """Fetch view count, likes, comments, and duration for a list of videos"""
    # Videoni uzeltiin too, laikuud, setgegdeluud, urgeljleh hugatsaag avah
    # Olon toonii videoni tootsoolson metrikuudiig tsugluulah
    stats = []
    
    # Process in batches of 50 (API limit)
    # API-n hetseeg ashiglaltiin hyzgaarlaltaas zailshiih 50 batcheer bulegleh
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        request = youtube.videos().list(
            part='statistics,contentDetails',
            id=','.join(batch)
        )
        response = request.execute()
        
        for item in response['items']:
            video_id = item['id']
            view_count = int(item['statistics'].get('viewCount', 0))
            like_count = int(item['statistics'].get('likeCount', 0))
            comment_count = int(item['statistics'].get('commentCount', 0))
            duration = item['contentDetails']['duration']
            
            stats.append({
                'video_id': video_id,
                'view_count': view_count,
                'like_count': like_count,
                'comment_count': comment_count,
                'duration': duration
            })
        
        # Avoid API quota issues
        # API-n quota hetruulehees zailshiih zoriulalt timesleep
        time.sleep(1)
    
    return stats

def collect_event_data(api_key, events, results_per_event=30):
    """Collect data for multiple events and combine into a dataset"""
    # Olon uil yavdliin medeelliig tsugluulj neg dataset bolgoh
    # API ashiglaj bugd uil yavdluudiin videoniig olj, negtgeh
    youtube = setup_youtube_api(api_key)
    all_videos = []
    
    for event in events:
        print(f"Collecting data for event: {event}")
        videos = search_event_videos(youtube, event, results_per_event)
        all_videos.extend(videos)
    
    # Create DataFrame
    # All videonii medeelelig agulsan DataFrame uusgeh
    videos_df = pd.DataFrame(all_videos)
    
    if videos_df.empty:
        print("No videos found!")
        return pd.DataFrame()
    
    # Get video IDs
    # Video bolgonii ID-g jagsaalt helbereer avah
    video_ids = videos_df['video_id'].tolist()
    
    # Get statistics for all videos
    # Buh videonii statistic medeelliig avah
    video_stats = get_video_statistics(youtube, video_ids)
    stats_df = pd.DataFrame(video_stats)
    
    # Merge data
    # Video bolon statistic medeelliig video_id-gaar niiluuleh
    merged_df = pd.merge(videos_df, stats_df, on='video_id')
    
    # Process dates and durations
    # Oguulsen ognoo, hugatsaag bolovsruulah
    merged_df['published_at'] = pd.to_datetime(merged_df['published_at'])
    
    # Use timezone-aware datetime for calculating days
    # Timezone medeeleltei ognoot ashiglaj video niitlegsneesee hoish kheden honog bolsniig tootsooloh
    now_with_tz = datetime.now(timezone.utc)
    merged_df['days_since_publishing'] = (now_with_tz - merged_df['published_at']).dt.days
    
    # Add engagement metrics
    # Oroltsoonii metrikuudiig nemeh
    merged_df['engagement_ratio'] = (merged_df['like_count'] + merged_df['comment_count']) / merged_df['view_count']
    
    # Safely calculate views per day (avoid division by zero)
    # Udur tutmiin uzeltiin toog ayulgui tootsooloh (tegloorh huvaaltaas zailshiih)
    merged_df['views_per_day'] = merged_df.apply(
        lambda row: row['view_count'] / max(row['days_since_publishing'], 1), axis=1
    )
    
    return merged_df

def save_data(data, filename):
    """Save the data to a CSV file"""
    # Medeelliig CSV file bolgon hadgalah
    data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def load_data(filename):
    """Load data from a CSV file"""
    # CSV fileaas medeelel unshih
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print(f"File {filename} not found!")
        return None