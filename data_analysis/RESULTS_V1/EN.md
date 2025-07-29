
## 0% accidentes - Perfect
![epsilon_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/epsilon.png)

## 5% accidentes - Good
![epsilon_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/epsilon.png)

## 0% accidentes - Perfect
![landings_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/landings.png)

## 5% accidentes - Good
![landings_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/landings.png)

## 0% accidentes - Perfect
![accidentes_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/crashes.png)

## 5% accidentes - Good
![accidentes_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/crashes.png)

## 0% accidentes - Perfect
![bonus_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/bonus.png)

## 5% accidentes - Good
![bonus_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/bonus.png)

## 0% accidentes - Perfect
![soft_accidentes_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/soft_crashes.png)

## 5% accidentes - Good
![soft_accidentes_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/soft_crashes.png)

## 0% accidentes - Perfect
![rAE_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/weighted_rAE.png)

## 5% accidentes - Good
![rAE_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/weighted_rAE.png)

## 0% accidentes - Perfect
![rC_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/weighted_rC.png)

## 5% accidentes - Good
![rC_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/weighted_rC.png)

## 0% accidentes - Perfect
![rL_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/weighted_rL.png)

## 5% accidentes - Good
![rL_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/weighted_rL.png)

## 0% accidentes - Perfect
![rME_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/weighted_rME.png)

## 5% accidentes - Good
![rME_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/weighted_rME.png)

## 0% accidentes - Perfect
![rSC_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/weighted_rSC.png)

## 5% accidentes - Good
![rSC_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/weighted_rSC.png)

## 0% accidentes - Perfect
![rCP_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/weighted_tCP.png)

## 5% accidentes - Good
![rCP_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/weighted_tCP.png)

---

## 🔍 Comparative Analysis

- **Crash rate, landings, and epsilon**: Behave similarly between models.
- **Key differences:**

| Metric       | Perfect (0%)        | Good (5%)              |
|--------------|---------------------|--------------------------|
| **Bonus**     | Min: -400 pts       | Min: -1200 pts           |
| **rL**        | Peaks at ~6000 pts  | Stabilizes around ~3500 pts |
| **rME**       | Dips mid-training   | Dips at the end          |

---

## 🔧 Code Observations

```python
if wrong_direction:
    return {'reward': -100, 'WrongDirection': True}
```

---

## 💡 Learning Hypothesis

1. The model starts with **high randomness** (high epsilon).
2. As it begins to land correctly (by chance or early learning), **epsilon decays quickly**.
3. At that point, **wrong_direction** penalties are crucial to prevent excessive ascents.
4. Learn to landing whit constant hight scores and no crashes.
---

## 🛠 Optimization Possibilities

- **Improve altitude learning**: Help the agent avoid rising too quickly or unnecessarily.
- **Early stopping criteria**: Interrupt training if ~750 steps pass without altitude penalties.
- **Penalize lateral thrusters**: Add stronger penalties for unnecessary side thrusts after landing.

---

## 🚀 What the Best Models Learn

- **Safety**: No crashes.
- **Consistency**: Reliable, repeated landings.
- **Stability**: No destabilizing rotation.
- **Efficiency**: No redundant steps or wasteful behavior.

> ⚠️ *Room for improvement*: Lateral thrusters are sometimes used after safe landings. A **stronger penalty** post-landing could help reduce this.