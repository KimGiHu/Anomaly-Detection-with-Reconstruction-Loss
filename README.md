# Downtime-Detection-Result

1. 가속기 정상운행 신호에서 이상치를 Detection한 결과를 케이스별로 모아놓음.

2. 현재 발생하고 있는 이슈에 대한 데이터세트를 개발 중에 있음 (해결완료.) >> 고유채널별 스케일링 및 생성된 오류신호 복원과정이 잘 이루어지는 것을 확인하였음.

Convolutional Neural Network의 문제점 : overfitting이 되는 것과 같은 효과가 발생함.

개선방향 : 트랜스포머 기반의 모델과 representaiton learning 방향성을 접목하여 시도해보고 있음

- 체크포인트1 : MLP와 트랜스포머 신경망을 이용한 오토-인코더 모델 설계 --> 오버피팅 현상이 해소되면서, 복원신호와 이상신호와의 차이도 명시적으로 확인하기. (완료)

- 체크포인트2 : 일부 구간에서 정상신호 스케일링하기, 전 구간 및 특정 구간에서 랜덤 가우시안 노이즈 추가해서 복원손실값 비교. (완료)

- 체크포인트3 : 생성형 모델 이용하여 실제 이상신호와 유사한 데이터셋과 정상 데이터셋을 분류해내 테스트. (완료)

연구성과 : 

1. 저널 게재 1건
Anomaly Detection using Pulse Reconstruction with Transformer-based VAE in the KOMAC High-power Systems, Journal of the Korean Physical Society(SCIE), published, April 2025, https://link.springer.com/article/10.1007/s40042-025-01339-0

2. 국제학술대회 발표 1건
Fault Detection using Pulse Reconstruction with CVAE in the KOMAC High-power Systems, The 26th International Conference on Accelerators and Beam Utilizations, Nov 2024


----------------------------------------------------------------------------------------------------------------------------------------
참고자료1 : https://medium.com/@hugmanskj/%ED%91%9C%ED%98%84%ED%95%99%EC%8A%B5-representation-learning-%EA%B0%9C%EC%9A%94-ea8d6252ea83

참고자료2 : https://seunghyun-lee.tistory.com/67

필수 라이브러리 : "requirements.txt"
