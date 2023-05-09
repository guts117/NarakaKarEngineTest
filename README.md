# NarakaKarEngineTest

**操作方法：**

>　WASD  移動

>　Space 上がる

>　Right CTRL　下がる

>　Spotlight ON/OFF

>　ESC 終了

>　X ２D流体マウス入力モード

>　Z 2D流体煙モード

**このサンプルーにて、追加した新規機能**

-２D流体シミュレーション
-RaymarchingとSDFをしようして、３Dテキスチャーのの中での3D流体シミュレーション

参考資料：

2D 
- GPU GEMS CHAPTER 38: 
https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu
- 2DFluidSimulation(2DシムとRK方式の参考)
https://github.com/cgurps/2DFluidSimulation

自分の追加の部分：
- RKに必要なヴェクターフィールドの情報をtextureArrayを使用して保持。
- 最大のヴェクターを持つセールの計算方法。
=======================================================================================================================================

3D
- GPU GEMS 3 CHAPTER 30:
https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids
- FLuid Simulation
https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf

自分の追加の部分：
- GPU　Gem　3　では、AABBボリュームの計算には、一つのPrePassが必要でした。
 今回は３Dキューブを作成して、そのキューブのローカルスペースで、３DTextureをサンプルしています。
 この方法だと、一つのシェーダーパスが要らなくなり、開始点がキューブの補間された頂点になって、Raymarchingが早く終了することが可能です。
- 2DでやったRK方式を使用して、ヴェクターフィールドの計算
- Computeシェーダを使用したシミュレーション。

=======================================================================================================================================

SDF + Raymarching 
- https://www.youtube.com/watch?v=62-pRVZuS5c&ab_channel=InigoQuilez
- https://iquilezles.org/articles/distfunctions/


その他
Runge Kutta Method:
https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
https://byjus.com/maths/runge-kutta-rk4-method/
