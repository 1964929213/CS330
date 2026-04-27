# 导航

这里先放 0416 那版对齐出来的数据，导航那边可以直接拿来做路网。

```text
navigation/baseline_0416_1/
  floorplan.jpg
  data/reference_nodes.csv
  data/destinations_template.csv
  data/edges_template.csv
  topometric/reference_points_2d.csv
  topometric/transform.json
  topometric/correspondences.csv
```

`reference_nodes.csv` 是参考图投到平面图上的点。可以先从这些点里挑一部分当路网节点。

大概需要接着做的就是：

1. 选一些合适的节点；
2. 在 `edges_template.csv` 里填哪些节点之间能走；
3. 在 `destinations_template.csv` 里填教室、楼梯、电梯这些目的地；
4. 跑 Dijkstra 或者 A*；
5. 把重定位出来的 `(x, y)` 接到最近的路网节点。

第一版先手工填边就行，不用一开始就做得很复杂。后面如果有时间，再考虑用墙体或者 boundary 自动判断能不能连边。

`reference_nodes.csv` 的列是：

```text
node_id,image_name,x,y,is_control_point
```

这里的 `x/y` 是平面图像素坐标。
