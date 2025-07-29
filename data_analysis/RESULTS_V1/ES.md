
## 0% accidentes - Perfecto
![epsilon_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/epsilon.png)

## 5% accidentes - Bueno
![epsilon_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/epsilon.png)

## 0% accidentes - Perfecto
![landings_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/landings.png)

## 5% accidentes - Bueno
![landings_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/landings.png)

## 0% accidentes - Perfecto
![accidentes_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/crashes.png)

## 5% accidentes - Bueno
![accidentes_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/crashes.png)

## 0% accidentes - Perfecto
![bonus_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/bonus.png)

## 5% accidentes - Bueno
![bonus_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/bonus.png)

## 0% accidentes - Perfecto
![soft_accidentes_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/soft_crashes.png)

## 5% accidentes - Bueno
![soft_accidentes_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/soft_crashes.png)

## 0% accidentes - Perfecto
![rAE_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/weighted_rAE.png)

## 5% accidentes - Bueno
![rAE_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/weighted_rAE.png)

## 0% accidentes - Perfecto
![rC_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/weighted_rC.png)

## 5% accidentes - Bueno
![rC_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/weighted_rC.png)

## 0% accidentes - Perfecto
![rL_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/weighted_rL.png)

## 5% accidentes - Bueno
![rL_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/weighted_rL.png)

## 0% accidentes - Perfecto
![rME_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/weighted_rME.png)

## 5% accidentes - Bueno
![rME_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/weighted_rME.png)

## 0% accidentes - Perfecto
![rSC_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/weighted_rSC.png)

## 5% accidentes - Bueno
![rSC_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/weighted_rSC.png)

## 0% accidentes - Perfecto
![rCP_A](../../screenshoots/metadata_bonus321_08_10_PERFECT.pth/weighted_tCP.png)

## 5% accidentes - Bueno
![rCP_B](../../screenshoots/metadata_bonus368_52_10_GOOD.pth/weighted_tCP.png)


---

## 🔍 Análisis Comparativo

- **Tasa de accidentes, aterrizajes y epsilon**: Similares entre ambos modelos.
- **Diferencias clave:**

| Métrica         | Perfecto (0%)        | Bueno (5%)              |
|----------------|-----------------------|--------------------------|
| **Bonus**       | Mínimo: -400 pts     | Mínimo: -1200 pts       |
| **rL**          | Hasta 6000 pts       | Estable ~3500 pts       |
| **rME**         | Negativo a mitad     | Negativo al final       |

---

## 🔧 Observaciones del Código

```python
if wrong_direction:
    return {'reward': -100, 'WrongDirection': True}
```

---

## 💡 Hipótesis de Aprendizaje

1. **Exploración inicial** con epsilon alto.
2. Cuando empieza a aterrizar, el **epsilon cae**.
3. La penalización por `wrong_direction` es crucial para evitar ascensos excesivos en este punto.
4. Aprende a aterrizar de forma constante con alta puntuación.
---

## 🛠 Posibilidades de Optimización 

- **Mejorar el aprendizaje de altitud** para evitar ascensos innecesarios.
- **Criterio de parada temprana**: detener si no hay penalizaciones tras ~750 pasos.
- **Penalizar motores laterales** tras aterrizaje si se usan sin necesidad.

---

## 🚀 Lo que aprenden los mejores modelos

- **Seguridad**: No tienen accidentes.
- **Consistencia**: Aterrizan siempre.
- **Estabilidad**: Sin rotaciones peligrosas.
- **Eficiencia**: Sin pasos redundantes.

> ⚠️ *Área a mejorar*: Uso innecesario de motores laterales tras el aterrizaje. Se recomienda **aumentar la penalización**.



