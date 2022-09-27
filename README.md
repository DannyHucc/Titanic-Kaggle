# 鐵達尼號(Titanic)生存率預測

## 壹、簡介

### 一、項目背景：
#### 泰坦尼克號的沉沒是歷史上最著名的沉船事件之一。 1912 年 4 月 15 日，鐵達尼號在與冰山相撞後沉沒。不幸的是，船上的每個人都沒有足夠的救生艇，導致 2224 名乘客和船員中有 1502 人死亡。
#### 雖然生存中涉及到一些運氣因素，但似乎有些人比其他人更有可能生存。
### 二、目的：
#### 使用機器學習創建一個模型，預測什麼樣的人更有可能在鐵達尼號沉船中生存下來。

### 三、數據採集：
#### Kaggle泰坦尼克號項目頁面：https://www.kaggle.com/c/titanic
#### 在Data中下載數據：
#### train.csv： 訓練數據集，用於模型的構建
#### test.csv ：測試數據集，無生存信息
#### gender_submission.csv ：提交的結果文件示例

### 四、 數據說明：
#### 數據內容：
| 	Index	 | 	Meaning	 |
| -------- | -------- |
| 	PassengerID	 | 	乘客編號	 |
| 	Survived	 | 	生存與否（1-生存，0-死亡）	 |
| 	Pclass	 | 	船艙等級（1/2/3等艙）	 |
| 	Name	 | 	乘客姓名	 |
| 	Sex	 | 	性別（male-男性；female-女性）	 |
| 	Age	 | 	年齡	 |
| 	Parch	 | 	父母/小孩人數	 |
| 	SibSp	 | 	兄弟姐妹/配偶人數	 |
| 	Ticket	 | 	船票編號	 |
| 	Fare	 | 	票價	 |
| 	Cabin	 | 	船艙號	 |
| 	Embarked	 | 	登船地(C = Cherbourg, Q = Queenstown, S = Southampton)	 |

#### Pclass(船艙等級)：
| 	Index	 | 	Meaning	 |
| -------- | -------- |
| 	1	 | 	頭等	 |
| 	2	 | 	二等	 |
| 	3	 | 	三等	 |


#### Embarked(登船港口)：
| 	Index	 | 	Meaning	 |
| -------- | -------- |
| 	C	 | 	C=Cherbourg(法國 瑟堡市)	 |
| 	Q	 | 	Q=Queenstown(愛爾蘭 昆士敦)	 |
| 	S	 | 	S=Southampton(英國南安普頓)	 |

-----

## 贰、Exploratory Data Analysis (EDA) & Feature Engineering
### 一、缺失值查看：
#### 利用.isnull().sum()查看缺失值，確認Survived有 418 個缺失值、Age有263個缺失值、Fare有1個缺失值、Cabin有1014 個缺失值、Embarked有2個缺失值。
![](image/1.png)
#### 各資料的缺失值 - (圖1)

-----

### 二、Pclass(艙位)與Survived(生存率)的關係：
#### 使用sns.barplot()繪製Pclass(艙位)與Survived(生存率)的關係，可以看出頭等艙的生存率最高，反映出身分地位越高，能享有的服務品質會更好，遇難時會有一定優勢。
![](image/2.png)
#### 繪製 Pclass(艙位)與Survived(生存率)的關係 - (圖2)

-----

### 三、Sex(性別)與Survived(生存率)的關係：
#### 使用sns.barplot()繪製Sex(性別)與Survived(生存率)的關係，可以看出女性的生存率明顯高於男性，說明歐洲的人們普遍秉持著女士優先的傳統。
![](image/3.png)
#### 繪製Sex(性別)與Survived(生存率)的關係 - (圖3)

-----

### 四、SibSp(兄弟姐妹/配偶人數)與 Survived(生存率)的關係：
#### 使用sns.barplot()繪製SibSp(兄弟姐妹/配偶人數)與Survived(生存率)，可以看出人數少的生存率較高。
![](image/4.png)
#### 繪製SibSp(兄弟姐妹/配偶人數)與Survived(生存率)的關係 - (圖4)

-----

### 五、Parch(父母/小孩人數)與Survived(生存率)的關係：
#### 使用sns.barplot()繪製Parch(父母/小孩人數)與Survived(生存率)，可以看出人數少的生存率較高。
![](image/5.png)
#### 繪製Parch(父母/小孩人數)與Survived(生存率)的關係 -  (圖5)

-----

### 六、Embarked(登船地)缺失值處理
#### 使用.value.counts()計算Embarked(登船地)的’S’、’C’、’Q’資料數量，確認’S’最多，並將2個缺失值填入'S'。
![](image/6.png)
#### Embarked(登船地)’S’、’C’、’Q’資料數量 - (圖6)

-----

### 七、Embarked(登船地)與Survived(生存率)的關係：
#### 使用sns.barplot()繪製Embarked(登船地)與Survived(生存率)，可以看出Embarked(登船地)’C’的生存率最高。
![](image/7.png)
#### 繪製Embarked(登船地)與Survived(生存率)的關係 - (圖7)

-----

### 八、Fare(票價)缺失值處理：
#### 使用sns.barplot()查看Pclass(船艙)和Embarked(登船地)與Fare(票價)的關係。
![](image/8.png)
#### Pclass(船艙)的頭等艙與Fare(票價)最有關係 - (圖8)
![](image/9.png)
#### Embarked(登船地)’S’與Fare(票價)最有關係 - (圖9)
#### 將Pclass(船艙)為'1'且Embarked(登船地)為'S'的Fare(船票)的資料的平均值，填補到Fare(船票)的缺失值資料裡。

-----

### 九、Fare(票價)與Survived(生存率)的關係：
#### 使用sns.FacetGrid()繪製Fare(票價)與Survived(生存率)的關係。
![](image/10.png)
#### 繪製Fare(票價)與Survived(生存率)的關係 - (圖10)

-----

### 十、Name(姓氏)資料處理：
#### 使用.split()將名字分割為姓氏，新增 Title 欄位並使用.map()依照姓氏的相關意思進行分類，將相同姓氏做資料合併。

![](image/11.png)
#### 姓氏的數量 - (圖11)

#### 姓氏的相關意思：
| 	姓氏	 | 	意思	 |
| 	--------	 | 	--------	 |
| 	Mr.	 | 	Mister,先生	 |
| 	Miss	 | 	對未婚婦女用	 |
| 	Mlle	 | 	小姐	 |
| 	Master	 | 	傭人對未成年男少主人的稱呼	 |
| 	jonkheer	 | 	貴族	 |
| 	Mme	 | 	Madame的簡寫,夫人	 |
| 	Ms.或Mz	 | 	稱呼婚姻狀態不明的婦女	 |
| 	Mrs.	 | 	Mistress,太太/夫人	 |
| 	Don	 | 	先生（至於男士名字前的尊稱）	 |
| 	The Countless	 | 	女伯爵	 |
| 	Dona	 | 	西班牙語對女子的稱謂,相當於Lady	 |
| 	Lady	 | 	女士,指成年女子	 |
| 	Capt	 | 	Captain,隊長	 |
| 	Sir	 | 	閣下、爵士、先生	 |
| 	Col	 | 	Colonel,上校	 |
| 	Major	 | 	有少校人意思	 |
| 	Dr.	 | 	Doctor,醫生/博士	 |
| 	Rev.	 | 	Reverend,用於基督教的牧師	 |

#### 分類方式：
| 	分類	 | 	姓氏	 |
| 	--------	 | 	--------	 |
| 	Mr	 | 	Mr	 |
| 	Miss	 | 	Mlle, Miss	 |
| 	Master	 | 	Master, Jonkheer	 |
| 	Mrs	 | 	Mme, Ms, Mrs	 |
| 	Royalty	 | 	Don, the Countess, Dona, Lady,Sir	 |
| 	Officer	 | 	Capt, Col, Major, Dr, Rev	 |

![](image/12.png)
#### Title欄位的姓氏數量 - (圖12)

-----

### 十一、 Title欄位與Survived(生存率)的關係
#### 使用sns.barplot繪製Title欄位的姓氏。
![](image/13.png)
#### 繪製Title欄位的姓氏 - (圖13)

-----

### 十二、Parch(父母/小孩人數)和SibSp(兄弟姐妹/配偶人數)資料處理
#### 新增familyNum欄位，將Parch(父母/小孩人數)和SibSp(兄弟姐妹/配偶人數)做資料合併。
#### 使用sns.barplot繪製familyNum與Survived(生存率)的關係，可以看出家族規模為2,3,4時，生存率是最高的。

![](image/14.png)
#### 繪製familyNum與Survived(生存率)的關係 - (圖14)

#### 新增 familySize 欄位，將家族規模生存率接近的資料分類：
#### 家族規模 1,7 生存率接近分為一類，以數字0表示；
#### 家族規模 2,3,4 生存率接近分為一類，以數字1表示；
#### 家族規模 5,6 生存率接近分為一類，以數字2表示。

-----

### 十三、familySize欄位與Survived(生存率)的關係
#### 使用sns.barplot繪製familySize與Survived(生存率)的關係，可以看出分類'0'的生存率最高。

![](image/15.png)
#### 繪製familySize與Survived(生存率)的關係 - (圖15)

-----

### 十四、 Age(年齡)缺失值處理
#### 可以看出0~10歲和15~40歲這兩個年齡層Survived(生存率)較大，遇難時優先保護小孩子是普遍的價值觀；一般出遊的人年紀落在15~40歲的人數會相對較多，所以Survived(生存率)有集中於15~40歲的趨勢。

![](image/16.png)
#### 繪製 Survived(生存率) 與 Survived(生存率)的關係- (圖16)

#### 將Age,Pclass,Title,TickCot,familyNum,Parch,SibSp 資料給值到變數 AgePre；使用get_dummies()，對變數AgePre進行類別變量轉換為標籤變量。
#### 新增變數AgeCorrDf設為DataFrame的二維資料結構；使用corr()計算AgePre的相關係數求取並給值到變數AgeCorrDf，查看 Age欄位，可以看出Parch、SibSp、Pclass的相關係數較高(取值接近-1，表示反相關，類似反比例函數，取值接近1，表正相關

![](image/17.png)
#### 查看AgeCorrDf的 Age欄位- (圖17)

#### 分別對Parch、SibSp、Pclass進行get_dummies()，然後給值到變數ParAge、SibAge、PclAge；將資料AgePre、ParAge、SibAge、PclAge 作合併到AgePre。

![](image/18.png)
#### 查看AgePre的欄位- (圖18)

#### 拆分AgePre資料的Age欄位，分為Features和Label，Features將Age欄位刪除，Label只取Age欄位。
#### 利用RF(RandomForestRegressor)建立模型，使用fit()進行資料預處理，使用score()進行評分；取出AgePre資料age為null的資料，使用drop()將age欄位刪除，製作生成Prediction使用的Features。
#### 取出AgePre資料age為null的資料，使用drop()將age欄位刪除，製作生成Prediction使用的Feature，使用predict()預測Age，將預測 Age 填入到缺失值裡。
