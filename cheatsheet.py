
----------------------------------------------------------Numpy array----------------------------




#-----1. Création d'un array numpy


# Import du module numpy sous l'alias 'np'
import numpy as np

# Création d'une matrice de dimensions 5x10 remplie de zéros
X = np.zeros(shape = (5, 10))

# Création d'une matrice à 3 dimensions 3x10x10 remplie de uns
X = np.ones(shape = (3, 10, 10))

# Création d'un array à partir d'une liste définie en compréhension
X = np.array([2*i for i in range(10)])    # 0, 2, 4, 6, ..., 18

# Création d'un array à partir d'une liste de listes
X = np.array([[1, 3, 3],
              [3, 3, 1],
              [1, 2, 0]])





#-----2. Indexation d'un array numpy


# Création d'une matrice de dimensions 10x10 remplie de uns
X = np.ones(shape = (10, 10))

# affichage de l'élément à l'index (4, 3)
print(X[4, 3])

> # assignation de la valeur -1 à l'élément d'index (1, 5)
X[1, 5] = -1


# Création de 2 arrays à 8 éléments
X = np.array([3, -7, -10, 3, 6, 5, 2, 9])

y = np.array([0, 1, 1, 1, 0, 1, 0, 0])

#Ces fonctions peuvent être appliquées sur tous les arrays numpy, peu importe leurs dimensions :


X = np.array([i/100 for i in range(100)])  # 0, 0.01, 0.02, 0.03, ..., 0.98, 0.99

# Calcul de l'exponentielle de x pour x = 0, 0.01, 0.02, 0.03, ..., 0.98, 0.99
exp_X = np.exp(X)


    (a) Définir une fonction f prenant en argument un array X et permettant de calculer en une seule ligne de code la fonction suivante :

    f(x)=exp(sin(x)+cos(x))

 X = np.array([i/100 for i in range(100)])

# Définition de la fonction f
def f(X):
    return np.exp(np.sin(X) + np.cos(X))

# Calcul de f(X)
resultat = f(X)

# On arrondi le résultat à 2 décimales
arrondi = np.round(resultat, decimals = 2)

# Affichage des 10 premiers éléments du résultat arrondi
print(arrondi[:10])



----------------------------------------------Numpy manipulation des données---------------------------------

# Création d'un array de dimension 3x3
X = np.array([[-1, 0, 30],
              [-2, 3, -5],
              [5, -5, 10]])

# On assigne à tous les éléments négatifs la valeur 0
X[X<0] = 0

# Affichage de la matrice modifiée
print(X)
>>> [[ 0  0 30]
>>>  [ 0  3  0]
>>>  [ 5  0 10]]

# On assigne la valeur -1 aux éléments de X pour lesquels la valeur de y à l'indice correspondant vaut 1
X[y == 1] = -1

# Affichage de X
print(X)
>>> [3 -1 -1 -1 6 -1 2 9]

# Affichage des éléments de X pour lesquels la valeur de y à l'indice correspondant vaut 0
print(X[y == 0])
>>> [3 6 2 9]



-------------#3. Redimensionnement d'un array

# Création d'un array à partir d'une liste à 10 éléments
X = np.array([i for i in range(1, 11)])   # 1, 2, ..., 10

# Affichage des dimensions de X
print(X.shape)
>>> (10,)

# Affichage de X
print(X)
>>> [1  2  3  4  5  6  7  8  9 10]

# Reshaping de l'array en une matrice à 2 lignes et 5 colonnes
X_reshaped = X.reshape((2, 5))

# Affichage du nouvel array
print(X_reshaped)
>>> [[ 1  2  3  4  5]
>>>  [ 6  7  8  9 10]]




--------------#4. Concaténation d'arrays

# Création de deux arrays de 3 lignes et 2 colonnes
# Le premier est rempli de 1
X_1 = np.ones(shape = (3, 2))
print(X_1)
>>> [[1. 1.]
>>>  [1. 1.]
>>>  [1. 1.]]

# Le deuxième est rempli de 0
X_2 = np.zeros(shape = (3, 2))
print(X_2)
>>> [[0. 0.]
>>>  [0. 0.]
>>>  [0. 0.]]

# Concaténation des deux arrays sur l'axe des lignes
X_3 = np.concatenate([X_1, X_2], axis = 0)
print(X_3)
>>> [[1. 1.]
>>>  [1. 1.]
>>>  [1. 1.]
>>>  [0. 0.]
>>>  [0. 0.]
>>>  [0. 0.]]

# Concaténation des deux arrays sur l'axe des colonnes
X_4 = np.concatenate([X_1, X_2], axis = 1)
print(X_4)
>>> [[1. 1. 0. 0.]
>>>  [1. 1. 0. 0.]
>>>  [1. 1. 0. 0.]]



--------------------------------Opération sur les arrays Numpy--------------------



1. Opérateurs arithmétiques


# Création de deux arrays à 2 valeurs
a = np.array([4, 10]) 
b = np.array([6, 7])   

# Multiplication entre deux arrays
print(a * b)
>>> [24 70]

# Création de deux arrays de dimensions 2x2
M = np.array([[5, 1],
              [3, 0]])

N = np.array([[2, 4],
              [0, 8]])

# Produit matriciel entre les deux arrays
print(M.dot(N))
>>> [[10 28]
>>>  [ 6 12]]



4. Les méthodes statistiques

A = np.array([[1, 1, 10],
              [3, 5, 2]])

# Calcul de la moyenne sur TOUTES les valeurs de X
print(A.mean())
>>> 3.67

# Calcul de la moyenne sur les COLONNES de X
print(A.mean(axis = 0))
>>> [2. 3. 6.]

# Calcul de la moyenne sur les LIGNES de X
print(A.mean(axis = 1))
>>> [4. 3.33]

#calculer la normalisation Min-Max très rapidement à l'aide des méthodes min et max et du broadcasting :

X_tilde = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))

print(X_tilde)
>>> [[1.         1.        ]
>>>  [0.4        0.13043478]
>>>  [0.         0.        ]]

#Pour créer la liste [0.01, 0.02, ..., 0.13, 0.14, 0.15], vous pouvez utiliser la fonction np.linspace dont la signature est similaire à la fonction range :

print(np.linspace(start = 0.01, stop = 0.15, num = 15))
>>> [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1  0.11 0.12 0.13 0.14 0.15]


------------------------------------Chargement et première exploration sur un jeu de données avec Pandas--------------------

2. Création d'un DataFrame à partir d'un array NumPy


# Création d'un array NumPy 
array = np.array([[1, 2, 3, 4], 
                  [5, 6, 7, 8], 
                  [9, 10, 11, 12]])

# Instanciation d'un DataFrame 
df = pd.DataFrame(data = array,                 # Les données à mettre en forme
                  index = ['i_1', 'i_2', 'i_3'],  # Les indices de chaque entrée
                  columns = ['A', 'B', 'C', 'D']) # Le nom des colonnes



3. Création d'un DataFrame à partir d'un dictionnaire

# Création d'un dictionnaire
dictionnaire = {'A': [1, 5, 9], 
                'B': [2, 6, 10],
                'C': [3, 7, 11],
                'D': [4, 8, 12]}

# Instanciation d'un DataFrame 
df = pd.DataFrame(data = dictionnaire,
                  index = ['i_1', 'i_2', 'i_3'])



4. Création d'un DataFrame à partir d'un fichier de données

pd.read_csv(filepath_or_buffer , sep = ',', header = 0, index_col = 0 ... )
#exepmle
mon_dataframe=pd.read_csv("transactions.csv"  , sep = ',', header = 0, index_col = 0)




6. Visualisation d un DataFrame: méthode head, attributs columns et shape./
mon_dataframe.head(10)

#Afficher les dimensions du DataFrame transactions ainsi que le nom de la 5ème colonne. Rappelez-vous qu'en Python les indices commencent à 0.

print(transactions.shape)
print(transactions.columns[4])

>>> (23053, 9)
>>>  Qty



7. Sélection de colonnes d'un DataFrame'

# Affichage de la colonne 'cust_id' 
print(transactions['cust_id'])

# Extraction des colonnes 'cust_id' et 'Qty' de transactions
cust_id_qty = transactions[["cust_id","Qty"]]




8. Sélection de lignes d'un DataFrame: méthodes loc et iloc'

# On récupère la ligne d'indice 80712190438 du DataFrame num_vars
print(num_vars.loc[80712190438])

>>>                 Rate    Tax  total_amt
>>> transaction_id                         
>>> 80712190438    -772.0  405.3    -4265.3
>>> 80712190438     772.0  405.3     4265.3

# On récupère les lignes d'indice 80712190438, 29258453508 et 51750724947 du DataFrame transactions
transactions.loc[[80712190438, 29258453508, 51750724947]]

# On extrait les colonnes 'Tax' et 'total_amt' des lignes d'indices 80712190438 et 29258453508
transactions.loc[[80712190438, 29258453508], ['Tax', 'total_amt']]

# Extraction des 4 premières lignes et des 3 premières colonnes de transactions
transactions.iloc[0:4, 0:3]



9. Indexation Conditionnelle d'un DataFrame'


# On séléctionne les lignes du DataFrame df pour lesquelles la colonne 'col 2' vaut 3. 
df[df['col 2'] == 3]

df.loc[df['col 2'] == 3]

df[df['col 2'] == 3] ne renvoie qu'une copie'
Si nous souhaitons assigner une nouvelle valeur à ces entrées, il faut absolument utiliser la méthode loc. 


# Création de transactions_eshop par indexation conditionnelle
transactions_eshop = transactions.loc[transactions['Store_type'] == 'e-Shop']

# Extraction des colonnes cust_id' et 'tran_date'
transactions_id_date = transactions_eshop[['cust_id', 'tran_date']]

# Affichage des 5 premières lignes de transactions_id_date
transactions_id_date.head()



10. Rapide étude statistique des données d'un DataFrame'

# Insérez votre code ici 
#La méthode describe d'un DataFrame retourne un résumé des statistiques descriptives (min, max, moyenne, quantiles,..) de ses variables quantitatives.
transactions.describe()

transactions[transactions['total_amt']  > 0].describe()

# Le montant moyen des transactions de montant positif est de 2608€, 500 euros
# de plus que la moyenne que nous avions obtenue précédemment.





Conclusion et récap

    La classe DataFrame du module pandas sera votre structure de données de choix pour explorer, analyser et traiter des bases de données.

    Dans cette brève introduction, vous avez appris à :

            Créer un DataFrame à partir d'un array numpy et d'un dictionnaire à l'aide du constructeur pd.DataFrame.'

            Créer un DataFrame à partir d'un fichier .csv à l'aide de la fonction pd.read_csv.

            Visualiser les premières et dernières lignes d'un DataFrame à l'aide des méthodes head et tail.

            Sélectionner une ou plusieurs colonnes d'un DataFrame en renseignant leurs noms entre crochets comme pour un dictionnaire.'

            Sélectionner une ou plusieurs lignes d'un DataFrame en renseignant leur indice à l'aide des méthodes loc et iloc.

            Sélectionner les lignes d'un DataFrame qui vérifient une condition spécifique à l'aide de l'indexation conditionnelle.'

            Effectuer une rapide étude statistique des variables quantitatives d'un DataFrame à l'aide de la méthode describe


----------------------------------------Data cleaning : Nettoyage des Données et Gestion des NAs --------------------------------------

1. Nettoyage d'un jeu de données'



        Gestion des doublons (méthodes duplicated et drop_duplicates).

        Modification des éléments d'un DataFrame (méthodes replace, rename et astype).'

        Opérations sur les valeurs d'un DataFrame (méthode apply et clause lambda).'


# On repère les lignes contenant des doublons
df.duplicated()

>>> 0  False
>>> 1  False
>>> 2  False
>>> 3  True


# Pour calculer la somme de booléens, on considère que True vaut 1 et False vaut 0.
print(df.duplicated().sum())
>>> 1

# On ne garde que la première occurence du doublon
df_first = df.drop_duplicates(keep = 'first')

# On ne garde que la dernière occurrence du doublon
df_last = df.drop_duplicates(keep = 'last')

# On ne garde aucun doublon
df_false = df.drop_duplicates(keep = False)

odification des éléments d'un DataFrame (méthodes replace, rename et astype)'

# Création du dictionnaire associant les anciens noms aux nouveaux noms de colonnes
dictionnaire = {'ancien_nom1': 'nouveau_nom1',
                'ancien_nom2': 'nouveau_nom2'}

# On renomme les variables grâce à la méthode rename
df = df.rename(dictionnaire, axis = 1)

# Méthode 1 : Création d'un dictionnaire puis appel à la méthode astype du DataFrame
dictionnaire = {'col_1': 'int',
                'col_2': 'float'}
df = df.astype(dictionnaire)

# Méthode 2 : Séléction de la colonne puis appel à la méthode astype d'une Series
df['col_1'] = df['col_1'].astype('int')


Opérations sur les valeurs d'un DataFrame (méthode apply et fonctions lambda)'

# Somme des lignes pour chaque COLONNE de df
 df_lines = df.apply(np.sum, axis = 0)

 # Somme des colonnes pour chaque LIGNE de df
 df_columns = df.apply(np.sum, axis = 1)


#Utilisation de lambda  

transactions['day'] = transactions['tran_date'].apply(lambda date: date.split('-')[0])

# Calcul du prix unitaire d'une produit
transactions.apply(lambda row: row['total_amt'] / row['qty'], axis = 1)

2. Gestion des valeurs manquantes



        La détection des valeurs manquantes (méthodes isna et any).

        Le remplacement de ces valeurs (méthode fillna).

        La suppression des valeurs manquantes (méthode dropna).




        La méthode any avec son argument axis permet de déterminer quelles colonnes (axis = 0) ou quelles lignes (axis = 1) contiennent au moins une valeur manquante.

        La méthode sum compte le nombre de valeurs manquantes par colonne ou lignes (en spécifiant l'argument axis). Il est possible d'utiliser d'autres méthodes statistiques comme mean, max, argmax, etc...


df.isna().any(axis = 0)

>>> Nom      True
>>> Pays     False
>>> Age      True

# On détecte les LIGNES contenant au moins une valeur manquante
df.isna().any(axis = 1)

>>> 0     True
>>> 1    False
>>> 2    False

# On utilise l'indexation conditionnelle pour afficher les entrées
# contenant des valeurs manquantes

df[df.isna().any(axis = 1)]

# On compte le nombre de valeurs manquantes pour chaque COLONNE
c #Les fonctions isnull et isna sont strictement équivalentes

>>> Nom     1
>>> Pays    0
>>> Age     1

# On compte le nombre de valeurs manquantes pour chaque LIGNE
df.isnull().sum(axis = 1)

>>> 0    2
>>> 1    0
>>> 2    0

Remplacement des valeurs manquantes (méthode fillna)

 On remplace tous les NANs du DataFrame par des zéros
 df.fillna(0) 

# On remplace les NANs de chaque colonne numérique par la moyenne sur cette colonne
 df.fillna(df.mean())  # df.mean() peut être remplacée par n'importe quelle méthode statistique.

 # On remplace les NANs de 'prod_subcat_code' par -1
transactions['prod_subcat_code'] = transactions['prod_subcat_code'].fillna(-1)

# On détermine le mode de 'store_type'
store_type_mode = transactions['store_type'].mode()
print("La modalité la plus fréquente de 'store_type' est:", store_type_mode[0])

# On remplace les NANs de 'store_type' par son mode
transactions['store_type'] = transactions['store_type'].fillna(transactions['store_type'].mode()[0])

# On vérifie que ces deux colonnes ne contiennent plus de NANs
transactions[['prod_subcat_code', 'store_type']].isna().sum()

Suppression des valeurs manquantes (méthode dropna)

 dropna(axis, how, subset, ..)



        how = 'any': On supprime la ligne (ou colonne) si elle contient au moins une valeur manquante.

        how = 'all' : On supprime la ligne (ou colonne) si elle ne contient que des valeurs manquantes.

 Exemple :

# On supprime toutes les lignes contenant au moins une valeur manquante
df = df.dropna(axis = 0, how = 'any')

# On supprime les colonnes vides 
df = df.dropna(axis = 1, how = 'all') 

# On supprime les lignes ayant des valeurs manquantes dans les 3 colonnes 'col2','col3' et 'col4'
 df.dropna(axis = 0, how = 'all', subset = ['col2','col3','col4'])


Conclusion et récap

    Dans ce chapitre nous avons vu les méthodes essentielles du module pandas afin de nettoyer un dataset et gérer les valeurs manquantes (NaN).

    Cette étape de préparation d'un dataset est toujours la première étape d'un projet data.

    Concernant le nettoyage des données, nous avons ainsi appris à :

            Repérer et supprimer les doublons d'un DataFrame grâce aux méthodes duplicated et drop_duplicates.'

            Modifier les éléments d'un DataFrame et leur type à l'aide des méthodes replace, rename et astype.

            Appliquer une fonction à un DataFrame avec la méthode apply et la clause lambda.

    Concernant la gestion des valeurs manquantes, nous avons appris à :

            Les détecter grâce à la méthode isna suivie des méthodes any et sum.

            Les remplacer à l'aide de la méthode fillna et des méthodes statistiques.'

            Les supprimer grâce à la méthode dropna.




----------------------------------------Data processing --------------------------

Le preprocessing de données peut se résumer à l'utilisation de 4 opérations essentielles : filtrer, unir, ordonner et grouper.'

1. Filtrer un DataFrame avec les opérateurs binaires.

es opérateurs adaptés au filtrage sur plusieurs conditions sont les opérateurs binaires :

        L'opérateur 'et' : &  '

        L'opérateur 'ou' : |  '

        L'opérateur 'non' : - ' 

# Filtrage du DataFrame sur les 2 conditions précédentes
print(df[(df['annee'] == 1979) & (df['surface'] > 60)])

>>>           quartier  annee  surface
>>> 0  Champs-Elysées   1979       70

# Filtrage avec 'ou' du DataFrame sur les 2 conditions précédentes
print(df[(df['année'] > 1900) | (df['quartier'] == 'Père-Lachaise')])

>>>          quartier  annee  surface
>>> 0  Champs-Elysées   1979       70
>>> 2   Père-Lachaise   1935       55
>>> 3           Bercy   1991       30



# Filtrage du DataFrame sur les 2 conditions précédentes
print(df[-(df['quartier'] == 'Bercy')])

>>>          quartier  annee  surface
>>> 0  Champs-Elysées   1979       70
>>> 1          Europe   1850      110
>>> 2   Père-Lachaise   1935       55



2. Unir des Dataframes : fonction concat et méthode merge.



# On renomme la colonne 'customer_Id' en 'cust_id' pour faire la fusion
customer = customer.rename(columns = {'customer_Id':'cust_id'})

# Jointure à gauche entre transactions et customer sur la colonne 'cust_id'
fusion = transactions.merge(right = customer, on = 'cust_id', how = 'left')

# La fusion n'a produit aucun NaN
fusion.isna().sum()

# Les colonnes DOB, Gender, city_code ont bien été ajoutées à transactions
fusion.head()

On peut définir la colonne 'Nom' comme étant le nouvel index :

df = df.set_index('Nom')


# Nouvel index à utiliser
new_index = ['10000' + str(i) for i in range(6)]
print(new_index)
>>> ['100000', '100001', '100002', '100003', '100004', '100005']

# Utiliser un array ou une Series est équivalent
index_array = np.array(new_index)
index_series = pd.Series(new_index)


df = df.set_index(index_array)
df = df.set_index(index_series)



3. Trier et ordonner les valeurs d'un DataFrame : méthodes sort_values et sort_index.'



# On trie le DataFrame df sur la colonne 'Points_bonus'
df_sorted = df.sort_values(by = 'Points_bonus', ascending = True)

# On trie le DataFrame df par la colonne 'Points_bonus' puis en cas d'égalité, par la colonne 'Note'.
df_sorted = df.sort_values(by = ['Points_bonus', 'Note'], ascending = True)


# On définit la colonne 'Note' comme l'index de df
df = df.set_index('Note')

# On trie le DataFrame df selon son index
df = df.sort_index()



4. Grouper les éléments d'un DataFrame: méthodes groupby, agg et crosstab.'

Effectuer une opération groupby complexe grâce aux fonctions lambda et aux méthodes groupby et agg :

# Chargement des données dans 'covid_tests.csv'
covid_df = pd.read_csv("covid_tests.csv", sep = ';', index_col = 'patient_id')
covid_df.head()


# Croisement des résultats des tests avec la réalité
pd.crosstab(covid_df['test_result'], 
            covid_df['infected'])

# Le nombre de faux négatifs est de 3

pd.crosstab(covid_df['test_result'], 
            covid_df['infected'],
            normalize = 0)

# Le taux de faux positifs est d'environ 9% contre environ 91% de vrais positifs 

functions_to_apply = {
    'column1' : ['min', 'max'],
    'column2' : [np.mean, np.std],
    'column3' : lambda x: x.max() - x.min()
    }

df.groupby('column_to_group_by').agg(functions_to_apply)


 --------------------------------------------------Introduction au Machine Learning avec scikit-learn -----------------------

-----------------------------------------Régression linéaire 


1. Utilisation de scikit-learn pour la régression linéaire

# Variables explicatives
X = df.drop(['price'], axis = 1)

# Variable cible
y = df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)


from sklearn.linear_model import LinearRegression

# Instantiation du modèle
lr = LinearRegression()

# Entraînement du modèle
lr.fit(X_train, y_train)

# Prédiction de la variable cible pour le jeu de données TRAIN
y_pred_train = lr.predict(X_train)

# Prédiction de la variable cible pour le jeu de données TEST
y_pred_test = lr.predict(X_test)


Evaluation de la performance du modèle



    La fonction mean_squared_error de scikit-learn s'utilise ainsi :'

        mean_squared_error(y_true, y_pred)

    où :

            y_true correspond aux vraies valeurs de la variable cible.

            y_pred correspond aux valeurs prédites par notre modèle.


from sklearn.metrics import mean_squared_error

# Calcul de la MSE entre les valeurs de la variable cible du jeu de données train et la prédiction sur X_train
mse_train = mean_squared_error(y_train, y_pred_train)

# Calcul de la MSE entre les valeurs de la variable cible du jeu de données test et la prédiction sur X_test
mse_test = mean_squared_error(y_test, y_pred_test)

print("MSE train lr:", mse_train)
print("MSE test lr:", mse_test)


from sklearn.metrics import mean_absolute_error

# Calcul de la MAE entre les vraies valeurs de la variable cible du train et la prédiction sur X_train
mae_train = mean_absolute_error(y_train, y_pred_train)

# Calcul de la MAE entre les vraies valeurs de la variable cible du test et la prédiction sur X_test
mae_test = mean_absolute_error(y_test, y_pred_test)

print("MAE train lr:", mae_train)
print("MAE test lr:", mae_test)

mean_price = df['price'].mean()

print("\nRelative error", mae_test / mean_price)

# L'erreur moyenne est d'environ 20% du prix moyen, ce qui n'est pas optimal
# mais est quand même une bonne baseline pour tester des modèles plus avancés.


2. Surapprentissage sur les données avec un autre modèle de régression


from sklearn.ensemble import GradientBoostingRegressor

# Ces arguments ont été choisis pour surapprendre le plus possible
# Ne pas les utiliser en pratique
gbr = GradientBoostingRegressor(n_estimators = 1000,
                                max_depth = 10000,
                                max_features = 15,
                                validation_fraction = 0)

### MSE

# Calcul de la MSE entre les vraies valeurs de la variable cible du train et la prédiction sur X_train
mse_train_gbr = mean_squared_error(y_train, y_pred_train_gbr)

# Calcul de la MSE entre les vraies valeurs de la variable cible du test et la prédiction sur X_test
mse_test_gbr = mean_squared_error(y_test, y_pred_test_gbr)

print("MSE train gbr:", mse_train_gbr)
print("MSE test gbr:", mse_test_gbr, "\n")


### MAE

# Calcul de la MAE entre les vraies valeurs de la variable cible du train et la prédiction sur X_train
mae_train_gbr = mean_absolute_error(y_train, y_pred_train_gbr)

# Calcul de la MAE entre les vraies valeurs de la variable cible du test et la prédiction sur X_test
mae_test_gbr = mean_absolute_error(y_test, y_pred_test_gbr)

print("MAE train gbr:", mae_train_gbr)
print("MAE test gbr:", mae_test_gbr, "\n")

mean_price_gbr = df['price'].mean()

print("Relative error", mae_test_gbr / mean_price_gbr)


3. Pour aller plus loin : la Régression Linéaire Polynomiale


from widgets import polynomial_regression

polynomial_regression()


Conclusion et recap

    Dans ce cours, vous avez été introduits à la résolution d'un problème de régression grâce au Machine Learning.'

    Nous avons utilisé la bibliothèque scikit-learn pour instancier des modèles de régression comme LinearRegression ou GradientBoostingRegressor et appliquer des transformations sur les données comme l'extraction de variables polynomiales.

    Les différentes étapes que nous avons étudiées sont la base de toute résolution d'un problème de Machine Learning :'

            On prépare les données en séparant les variables explicatives de la variable cible.

            On sépare le jeu de données en deux (un jeu d'entraînement et un jeu de test) à l'aide de la fonction train_test_split du sous-module sklearn.model_selection.

            On instancie un modèle comme LinearRegression ou GradientBoostingRegressor grâce au constructeur de la classe.

            On entraîne le modèle sur le jeu de données d'entraînement à l'aide de la méthode fit.

            On effectue une prédiction sur les données de test grâce à la méthode predict.

            On évalue les performances de notre modèle en calculant l'erreur entre ces prédictions et les véritables valeurs de la variable cible des données de test.'

    L'évaluation de performances pour un modèle de régression se fait facilement grâce aux fonctions mean_squared_error ou mean_absolute_error du sous-module metrics de scikit-learn.'




---------------------- Introduction au Machine Learning avec Scikit-learn

-----------------------Partie II : Modèles simples de classification 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2 )


from sklearn.neighbors import KNeighborsClassifier

# Instanciation du modèle
knn = KNeighborsClassifier(n_neighbors = 6)

# Entraînement du modèle sur le jeu d'entraînement
knn.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred_test_knn = knn.predict(X_test)

# Affichage des 5 premières prédictions
print(y_pred_test_knn[:10])


Classification linéaire : Régression Logistique


# Importation de la classe LogisticRegression sous-module linear_model de sklearn
from sklearn.linear_model import LogisticRegression

# Instanciation du modèle
logreg = LogisticRegression()

# Entraînement du modèle sur le jeu d'entraînement
logreg.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred_test_logreg = logreg.predict(X_test)

# Affichage des 5 premières prédictions
print(y_pred_test_logreg[:10])


2. Evaluer la performance d'un modèle de classification'




        L'accuracy.'

        La précision et le rappel (precision et recall en anglais).

        Le score F1, ou F1-score en anglais.

from sklearn.metrics import confusion_matrix

# Calcul et affichage de la matrice de confusion
matrice_confusion = confusion_matrix(y_test, y_pred_test_knn)
print("Matrice de Confusion:\n",  matrice_confusion)

print("\nLe modèle knn a fait", matrice_confusion[0, 1], "Faux Positifs.")

# Calcul de l'accuracy, precision et rappel
(VN, FP), (FN, VP) = confusion_matrix(y_test, y_pred_test_knn)
n = len(y_test)

print("\nKNN Accuracy:", (VP + VN) / n)

print("\nKNN Précision:", VP / (VP + FP))

print("\nKNN Rappel:", VP / (VP + FN))


pd.crosstab(y_test, y_pred_test_knn, rownames=['Realité'], colnames=['Prédiction']


from sklearn.metrics import accuracy_score, precision_score, recall_score

# Calcul et affichage de la matrice de confusion
pd.crosstab(y_test, y_pred_test_logreg, rownames=['Realité'], colnames=['Prédiction'])

# Calcul de l'accuracy, precision et rappel
print("\nLogReg Accuracy:", accuracy_score(y_test, y_pred_test_logreg))

print("\nLogReg Précision:", precision_score(y_test, y_pred_test_logreg, pos_label = 'republican'))

print("\nLogReg Rappel:", recall_score(y_test, y_pred_test_logreg, pos_label = 'republican'))


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test_logreg))

print(classification_report(y_test, y_pred_test_knn))


from sklearn.metrics import f1_score

print("F1 KNN:", f1_score(y_test, y_pred_test_knn, pos_label = 'republican'))

print("F1 LogReg:", f1_score(y_test, y_pred_test_logreg, pos_label = 'republican'))


Conclusion et recap

    Scikit-learn propose de nombreux modèles de classification que l'on peut regrouper en deux familles :'

            Les modèles linéaires comme LogisticRegression.

            Les modèles non-linéaires comme KNeighborsClassifier.

    L'utilisation de ces modèles se fait de la même façon pour tous les modèles de scikit-learn :'

            Instanciation du modèle.

            Entraînement du modèle : model.fit(X_train, y_train).

            Prédiction : model.predict(X_test, y_test).

    La prédiction sur le jeu de test nous permet d'évaluer la performance du modèle grâce à des métriques adaptées.'

    Les métriques que nous avons vues s'utilisent pour la classification binaire et se calculent grâce à 4 valeurs :'

            Vrais Positifs : Prédiction = + | Réalité = +

            Vrais Négatifs : Prédiction = - | Réalité = -

            Faux Positifs : Prédiction = + | Réalité = -

            Faux Négatifs : Prédiction = - | Réalité = +

    Toutes ces valeurs peuvent se calculer à l'aide de la matrice de confusion générée par la fonction confusion_matrix du sous-module sklearn.metrics ou par la fonction pd.crosstab.'

    Grâce à ces valeurs, nous pouvons calculer des métriques comme :

            L'accuracy : La proportion d'observations correctement classifiées.

            La précision : La proportion de vrais positifs parmi toutes les prédictions positives du modèle.

            Le rappel : La proportion d'observations réellement positives qui ont été correctement classifiées positives par le modèle.'

    Toutes ces métriques peuvent s'obtenir à l'aide de la fonction classification_report du sous-module sklearn.metrics.

    Le F1-Score quantifie l'équilibre entre ces métriques, ce qui nous donne un critère fiable pour choisir le modèle le plus adapté à notre problème.'

