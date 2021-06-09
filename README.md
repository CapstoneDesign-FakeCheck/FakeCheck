# Fake Check
 요즈음 ‘딥페이크’로 인해 피해를 받는 사람들이 늘어나고 있다. 유명인의 얼굴부터 일반인들의 얼굴까지 가짜로 만들어내어 해당 사람이 전혀 하지 않은 행동을 하고 있는 이미지나 영상이 만들어지는 것이다. 
이를 방지하기 위해 딥페이크를 판별하는 여러 모델이 존재하지만, 판별기를 무력화 시키기 위해 제안된 Adversarial Attack에 아직은 취약하다.
deepfake로 인한 피해를 줄이기 위해서 모델 방해공격(adversarial attack)에도 robust한 모델을 구축하고, 애플리케이션을 구현하여 deepfake 방어에 대해 사람들의 접근성을 높이는 것을 목표로 본 과제를 수행했다.
<img height="200" src="https://github.com/CapstoneDesign-FakeCheck/FakeCheck/blob/master/pic/main_pic.jpg"/><br>
## DATA
* 유명인의 실제 영상과 deepfake가 적용된 영상으로 구성된 데이터 셋인 Celeb-DF 총 8,000장
* 직접 촬영한 팀원들의 얼굴 영상과 저작권이 없는 얼굴 이미지를 수집한 뒤 deepfake를 적용한 이미지 데이터셋 총 1,000장
* MTCNN 모델을 이용하여 얼굴 부분만 crop
| REAL | FAKE |
|:---:|:---:|
|<img height="200" src="https://github.com/CapstoneDesign-FakeCheck/FakeCheck/blob/master/pic/real.png"/>|<img height="200" src="https://github.com/CapstoneDesign-FakeCheck/FakeCheck/blob/master/pic/fake.jpg"/>|

## MODEL

## APP
Camera와 Gallery를 통해 사진을 받아온 뒤 MTCNN으로 얼굴 부분만 crop한다. 그 다음 tlite 모델의 input에 맞춰 이미지를 224*224로 resize하고 RGB채널을 BGR순서로 바꿔준다.<br>
<img height="300" src="https://github.com/CapstoneDesign-FakeCheck/FakeCheck/blob/master/pic/app_main.png"/>
<img height="300" src="https://github.com/CapstoneDesign-FakeCheck/FakeCheck/blob/master/pic/app_result.png"/>