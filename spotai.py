import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os, time
from stqdm import stqdm
load_dotenv()

cid = st.secrets['CLIENT_ID']
secret = st.secrets['CLIENT_SECRET']

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


class LoadPlaylist:
    def __init__(_self, plink1, plink2):
        _self.plink1 = plink1
        _self.plink2 = plink2
        _self.play1 = _self.generate_playlist_df(_self.plink1)
        _self.play2 = _self.generate_playlist_df(_self.plink2)

    @st.cache_data
    def generate_playlist_df(_self, plink):
        try:
            playlist_URI = plink.split("/")[-1].split("?")[0]
            column_list= ['uri','track_name','track_popularity', 'added_at', 'artist_name', 'artist_popularity', 'genre', 'artist_url']
            final_data = []

            for i in sp.playlist_tracks(playlist_URI)['items']:
                if i['track'] is not None:
                    artist_uri = i["track"]["artists"][0]["uri"]
                    artist_info = sp.artist(artist_uri)
                    artist_name = i["track"]["artists"][0]["name"]
                    artist_pop = artist_info["popularity"]
                    artist_genres = artist_info["genres"]

                    data_pos = [
                    i['track']['uri'],i['track']['name'], i["track"]["popularity"], i['added_at'].split('T')[0],
                    artist_name, artist_pop, ' '.join(artist_genres), i['track']['artists'][0]['external_urls']['spotify']
                    ]

                    final_data.append(data_pos)

            m_df = pd.DataFrame(final_data, columns = column_list)
            m_df['added_at'] = m_df['added_at'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())


            return m_df

        except Exception as e:
            return 'error'

    def similar_playlist_metrics(_self):

        try:
        
            m_sim_data1 = _self.play1['artist_name']+' '+ _self.play1['genre']
            m_sim_data2 = _self.play2['artist_name']+' '+ _self.play2['genre']

            m_simconcat1 = ' '.join(set(' '.join(m_sim_data1).lower().split()))
            m_simconcat2 = ' '.join(set(' '.join(m_sim_data2).lower().split()))

            cv = CountVectorizer()
            v1 = cv.fit_transform([m_simconcat1 if len(m_simconcat1)>len(m_simconcat2) else m_simconcat2])
            v2 = cv.transform([m_simconcat1 if len(m_simconcat1)<len(m_simconcat2) else m_simconcat2])

            return f'{round(cosine_similarity(v1,v2)[0][0]*100, 3)}' + ' %'
        except Exception as e :
            return 'Error calculating the similarity'

    def cross_recommend(_self):
        try:
            tags1 = (_self.play1['genre'] + ' ' + _self.play1['artist_name']).apply(lambda x: x.lower())
            tags2 = (_self.play2['genre'] + ' ' + _self.play2['artist_name']).apply(lambda x: x.lower())
            #picking tags of top 3 songs from playlist 1 and adding it to playlist 2 tags
            simseries = pd.concat([tags1[:3], tags2], axis=0, ignore_index=True)

            cv2 = CountVectorizer()
            m_vec = cv2.fit_transform(simseries).toarray()

            #calculating cosine similarity of each song with each other in playlist 2
            similarity = cosine_similarity(m_vec)
            top_n =10
            sm1 = similarity[0][2:].argsort()[-top_n:][::-1]
            sm2 = similarity[1][2:].argsort()[-top_n:][::-1]
            sm3 = similarity[2][2:].argsort()[-top_n:][::-1]


            # Combine the top indices from both arrays
            combined_indices = np.concatenate((sm1,sm2,sm3))

            # Remove duplicates if any (optional)
            combined_indices = np.unique(combined_indices)

            # Sort the combined indices to get the overall top indices
            overall_top_indices = combined_indices[np.argsort(-similarity[0][1:][combined_indices])][:top_n]
            recomff= _self.play2.iloc[[i for i in overall_top_indices]]

            finalf =  recomff[~recomff.isin(_self.play1)].iloc[:,1:].reset_index(drop=True)

            return finalf
        
        except Exception as e:
            return 'Error recommending songs, seems like there is error is loading the playlist'
    
    def get_stats(_self):
        try:
            index_list = ['oldest song','latest song', 'most popular song', 'most popular artist', 'most added artist']
            f1old = _self.play1.sort_values(by='added_at').iloc[0]['track_name'] #oldest song
            f2old = _self.play2.sort_values(by='added_at').iloc[0]['track_name'] #oldest song

            f1lat = _self.play1.sort_values(by='added_at', ascending=False).iloc[0]['track_name'] #latest song
            f2lat = _self.play2.sort_values(by='added_at', ascending=False).iloc[0]['track_name'] #latest song

            f1pop = _self.play1.sort_values(by='track_popularity', ascending=False).iloc[0]['track_name'] #most popular song
            f2pop= _self.play2.sort_values(by='track_popularity', ascending=False).iloc[0]['track_name'] #most popular song

            f1popart = _self.play1.sort_values(by='artist_popularity', ascending=False).iloc[0]['artist_name']  #most popular artist
            f2popart = _self.play2.sort_values(by='artist_popularity', ascending=False).iloc[0]['artist_name']  #most popular artist

            f1moccur = _self.play1['artist_name'].value_counts().index[0]+ f", {_self.play1['artist_name'].value_counts()[0]} times"
            f2moccur = _self.play2['artist_name'].value_counts().index[0]+ f", {_self.play2['artist_name'].value_counts()[0]} times"

            data = [[f1old, f2old], [f1lat, f2lat], [f1pop, f2pop], [f1popart, f2popart], [f1moccur, f2moccur]]

            stat_frame = pd.DataFrame(data, index=index_list, columns=['You', 'Your friend'])

            return stat_frame
        
        except Exception as e:
            return 'Error getting stats, seems like there is error is loading the playlist'
    
    def common_songs(_self):
        try :
            cmdf = pd.merge(_self.play1, _self.play2, how='inner', on=['uri'])
            return cmdf.iloc[:,1:8].reset_index(drop=True)
        except Exception as e:
            return 'Error finding common songs, seems like there is error is loading the playlist'
        
    def generate_plot(_self, link):
        try : 

            df = _self.generate_playlist_df(link)
            datedf = pd.DataFrame(df['added_at'].value_counts().reset_index())
            datedf.columns = ['date', 'count']

            st.bar_chart(datedf, x='date', y='count')
        except Exception as e:
            return 'Error generating stats, seems like there is error is loading the playlist'

if __name__ == '__main__':
    
    st.title('Spotafriend')
    st.caption('A fun tool to compare your spotify playlist with your friend !')
    st.write('---')
    
    friendl1 = st.text_input("Your spotify playlist link")
    friendl2 = st.text_input("Your friend's spotify playlist link")

    if st.button('Analyse the playlists'):
        st.write('---')
        lp = LoadPlaylist(friendl1, friendl2)
        score = lp.similar_playlist_metrics()
        st.header(score)
        st.caption(f"You and your friend's playlist is {score} similar")

        st.write('---')

        #Recommendation
        st.header('Recommendations')
        st.caption("Recommending songs that you don't have, from your friend's playlist that're similar from your top 3 songs")

        recomdf = lp.cross_recommend()
        st.write(recomdf)

        st.write('---')

        #stats
        st.header('Some stats')
        st.caption('Some cool stats based on both playlists')
        statdf = lp.get_stats()
        st.write(statdf)

        #common songs
        st.write('Common songs among you and your friend')
        comdf = lp.common_songs()
        try:

            if comdf.empty:
                st.warning('No common songs found between you and your friend')
            else:
                st.write(comdf)
        except Exception as e:
            pass

        col1, col2 = st.columns(2, gap='large')

        with col1:
            st.header('You')
            st.caption('Most songs added at dates')
            lp.generate_plot(friendl1)
        
        with col2:
            st.header('Your friend')
            st.caption('Most songs added at dates')
            
            lp.generate_plot(friendl2)

    











    
    

        
