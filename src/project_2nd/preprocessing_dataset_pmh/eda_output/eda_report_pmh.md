
# modeling_dataset.csv EDA 리포트 (pmh)

- 행 수: 1,889,582
- 열 수: 33
- 메모리 사용량: 694.0 MB

## 1. 기초 구조

```
                                    dtype
snapshot_date                         str
store_id                              str
industry_dae_code                     str
industry_group                        str
industry_jung_code                    str
industry_jung_name                    str
industry_code                         str
industry_name                         str
gu_name                               str
dong_code                             str
lng                               float64
lat                               float64
floor_category                        str
same_industry_count_300m            int64
total_count_300m                    int64
nearest_same_industry_distance_m  float64
dong_industry_count                 int64
coord_cluster_size                  int64
store_age_months                    int64
previously_transitioned             int64
keyword_growth_score              float64
korean_pop                        float64
foreign_long_pop                  float64
foreign_short_pop                 float64
total_pop_avg                     float64
foreign_short_ratio               float64
tourist_zone_candidate             object
industry_historical_rate          float64
dong_historical_rate              float64
dong_industry_historical_rate     float64
transitioned_next                    bool
fold                                int64
is_closed_next                      int64
```

### 결측치가 있는 컬럼

```
                        결측치 비율(%)
gu_name                       1.6
korean_pop                    1.6
foreign_long_pop              1.6
foreign_short_pop             1.6
total_pop_avg                 1.6
foreign_short_ratio           1.6
tourist_zone_candidate        1.6
```

- store_id+snapshot_date 중복: 0건
- 완전 중복 행: 0건

## 2. 타겟(is_closed_next) 분포

```
                    비율(%)
is_closed_next           
0               89.351507
1               10.648493
```

### fold별 표본수 / 폐업비율

```
         표본수      폐업비율
fold                  
0     377361  0.106333
1     379149  0.106088
2     380567  0.106567
3     375496  0.107301
4     377009  0.106141
```

### is_closed_next x transitioned_next 교차표

```
transitioned_next    False  True 
is_closed_next                   
0                  1670619  17751
1                   201212      0
```
![target_distribution_pmh](target_distribution_pmh.png)

## 3. 수치형 변수 분포

```
                                      count          mean           std          min           1%            5%           50%           95%            99%            max
lng                               1889582.0    126.992744      0.083503   126.768169   126.823646    126.847204    127.007654    127.125387     127.147567     127.182653
lat                               1889582.0     37.544481      0.048588    37.430448    37.461916     37.478158     37.541421     37.639956      37.666245      37.692604
same_industry_count_300m          1889582.0     19.922121     39.787822     0.000000     0.000000      0.000000      7.000000     77.000000     225.190000     466.000000
total_count_300m                  1889582.0    767.809834    558.230087     0.000000    49.000000    145.000000    610.000000   1892.000000    2621.000000    3781.000000
nearest_same_industry_distance_m  1889582.0    117.787713    210.007311     0.000000     0.000000      0.000000     54.677723    440.360578     956.115698    9220.676444
dong_industry_count               1889582.0     44.126410     76.539937     1.000000     1.000000      2.000000     18.000000    167.000000     438.000000     680.000000
coord_cluster_size                1889582.0     58.830580    121.576162     1.000000     1.000000      2.000000     17.000000    248.000000     647.000000    1220.000000
store_age_months                  1889582.0      9.671672      8.319141     0.000000     0.000000      0.000000      6.000000     24.000000      24.000000      24.000000
keyword_growth_score              1889582.0      0.001358      0.027610     0.000000     0.000000      0.000000      0.000000      0.000000       0.000000       1.842105
korean_pop                        1859401.0  31833.277073  18658.441727  3464.087606  7059.837187  11897.992859  26978.018713  67917.791977  101051.440314  103190.298593
foreign_long_pop                  1859401.0   1253.934107   1360.684041    52.497705   107.141189    173.114725    764.420635   4287.921983    6596.740771    8514.386657
foreign_short_pop                 1859401.0    929.907579   1982.731723     4.662842    14.895227     26.257768    181.237959   4843.155897    8233.336817   15157.378523
total_pop_avg                     1859401.0  34017.118759  20393.037813  3524.291229  7702.493583  12966.811971  28319.792544  70606.718524  108306.460431  108738.522246
foreign_short_ratio               1859401.0      0.021059      0.037605     0.000495     0.000999      0.001375      0.006644      0.099974       0.166810       0.266943
industry_historical_rate          1889582.0      0.106506      0.049474     0.022513     0.025289      0.044296      0.099974      0.211561       0.261137       0.853862
dong_historical_rate              1889582.0      0.093197      0.013031     0.053922     0.068582      0.074585      0.092221      0.114591       0.128172       0.210863
dong_industry_historical_rate     1889582.0      0.107412      0.065089     0.000000     0.000000      0.028662      0.098835      0.217949       0.296875       1.000000
```
![numeric_histograms_pmh](numeric_histograms_pmh.png)

## 4. 범주형 변수 분포


### industry_group (카디널리티=7)

```
                    건수
industry_group        
음식              689205
소매              547589
수리·개인           252911
교육              188177
예술·스포츠          100352
보건의료             71801
숙박               39547
```

### gu_name (카디널리티=25)

```
             건수
gu_name        
강남구      182634
송파구      117629
서초구      106020
마포구      104911
강서구       94759
영등포구      94208
중구        84670
관악구       77252
종로구       77246
강동구       70492
광진구       69043
동대문구      65746
구로구       65283
은평구       64009
노원구       63020
성북구       61483
양천구       61437
중랑구       59129
용산구       58819
성동구       56789
동작구       53274
서대문구      52640
금천구       49946
도봉구       41246
강북구       27716
```

### floor_category (카디널리티=5)

```
                    건수
floor_category        
1층              796393
결측              568011
2층이상            482845
기타               33488
지하                8845
```

- industry_jung_name 카디널리티: 53

- industry_code 카디널리티: 192

- dong_code 카디널리티: 428
![industry_group_counts_pmh](industry_group_counts_pmh.png)
![floor_category_counts_pmh](floor_category_counts_pmh.png)

## 5. 타겟과의 이변량 관계


### 수치형 변수: 타겟별 평균

```
is_closed_next                           정상(0)         폐업(1)
lng                                 126.992817    126.992133
lat                                  37.544611     37.543388
same_industry_count_300m             19.721867     21.602459
total_count_300m                    767.683421    768.870564
nearest_same_industry_distance_m    118.054055    115.552834
dong_industry_count                  43.566575     48.823987
coord_cluster_size                   59.096506     56.599199
store_age_months                     10.005835      6.867712
keyword_growth_score                  0.001344      0.001476
korean_pop                        31759.929121  32449.206884
foreign_long_pop                   1251.581994   1273.685671
foreign_short_pop                   930.436669    925.464610
total_pop_avg                     33941.947784  34648.357164
foreign_short_ratio                   0.021107      0.020654
industry_historical_rate              0.103781      0.129369
dong_historical_rate                  0.093050      0.094429
dong_industry_historical_rate         0.103646      0.139018
```

### 업종그룹(industry_group)별 폐업률

```
                     폐업률
industry_group          
교육              0.155981
음식              0.112488
소매              0.106459
예술·스포츠          0.100825
보건의료            0.096976
숙박              0.081726
수리·개인           0.062172
```

### 구(gu_name)별 폐업률

```
              폐업률
gu_name          
강남구      0.118039
강서구      0.117023
마포구      0.115107
송파구      0.112158
양천구      0.112033
구로구      0.110442
노원구      0.108775
성동구      0.108401
강동구      0.107899
관악구      0.107000
은평구      0.105454
금천구      0.104713
영등포구     0.104025
동작구      0.104009
서대문구     0.102641
서초구      0.101971
강북구      0.101746
성북구      0.100678
동대문구     0.100371
용산구      0.099934
중구       0.099835
광진구      0.099576
중랑구      0.099427
도봉구      0.098846
종로구      0.086296
```

### 층(floor_category)별 폐업률

```
                     폐업률
floor_category          
결측              0.123623
2층이상            0.110129
지하              0.098926
1층              0.092900
기타              0.088330
```

### 시점(snapshot_date)별 폐업률

```
                    폐업률
snapshot_date          
202312         0.072635
202406         0.133382
202412         0.111890
202506         0.101828
202512         0.112469
```
![closure_rate_by_group_and_time_pmh](closure_rate_by_group_and_time_pmh.png)

## 6. 수치형 변수 상관관계

![correlation_heatmap_pmh](correlation_heatmap_pmh.png)

### 타겟과의 상관계수 (절대값 순)

```
                                      corr
dong_industry_historical_rate     0.167629
industry_historical_rate          0.159533
store_age_months                 -0.116355
dong_historical_rate              0.032655
dong_industry_count               0.021187
same_industry_count_300m          0.014579
korean_pop                        0.011392
total_pop_avg                     0.010682
lat                              -0.007760
coord_cluster_size               -0.006336
foreign_long_pop                  0.005009
foreign_short_ratio              -0.003715
nearest_same_industry_distance_m -0.003674
lng                              -0.002528
keyword_growth_score              0.001474
foreign_short_pop                -0.000773
total_count_300m                  0.000656
```

## 7. 결측치 패턴

- 생활인구 결측 행: 30,181건 (1.60%)
- 결측이 발생하는 dong_code 수: 12 -> ['11230515', '11230533', '11305595', '11305603', '11305608', '11305615', '11305625', '11305635', '11530800', '11680675', '11740525', '11740526']

- nearest_same_industry_distance_m 결측 행: 0건 (0.00%)
  (동일업종 매장이 해당 스냅샷에서 자기 자신뿐인 경우와 일치하는지 same_industry_count_300m<=1 비교)
  -> 결측 행 중 same_industry_count_300m<=1 비율: nan%