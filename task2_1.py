import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#2.1
file_path = 'ratings.csv'
df = pd.read_csv(file_path)
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=100, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=50, axis=1)

ratings_matrix_filled = ratings_matrix.fillna(2.5)

R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=3)
S = np.diag(sigma)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
print("Matrix U:\n")
print(pd.DataFrame(U))

Ux  = U[:20, 0]
Uy = U[:20, 1]
Uz = U[:20, 2]
ax.scatter(Ux, Uy, Uz, color = 'red', marker='o')
plt.title('Users')
plt.show()

fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')
Vx = Vt[0, :20]
Vy = Vt[1, :20]
Vz = Vt[2, :20]
ax2.scatter(Vx, Vy, Vz, color = 'blue' , marker='o')
plt.title('Films')
plt.show()

#2.2
movies = pd.read_csv("movies.csv")
print("DATA BEFORE PREDICTION (REAL USER RATINGS)")
print(ratings_matrix.head())

all_user_predicted_ratings = U @ S @ Vt
all_user_predicted_ratings = np.dot(np.dot(U, S),Vt) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(all_user_predicted_ratings,columns=ratings_matrix.columns,index=ratings_matrix.index)

print("DATA AFTER PREDICTION (Усі клітинки заповнені: модель відновила і реальні, і невідомі рейтинги.")
print(preds_df.head())

only_predictions = preds_df.copy()
only_predictions[ratings_matrix.notna()] = np.nan

print("ONLY PREDICTED DATA (Якщо юзер дивився фільм - NaN, якщо ні то предікшн)")
print(only_predictions.head())

def recommend_movies(user_id, num_recs=10):
    preds = only_predictions.loc[user_id]
    top_ids = preds.sort_values(ascending=False).head(num_recs).index
    recs = movies[movies["movieId"].isin(top_ids)][["title", "genres"]]
    return recs

print("\nРекомендовані фільми для користувача 1:")
print(recommend_movies(1))

