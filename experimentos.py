"""Experimentos del taller con la red de Neurito (misma red que main.py, sin pygame)."""
import numpy as np

ACCIONES = ("COMER", "DORMIR", "JUGAR")


def etiquetas(X):
    # Política sintética: gana la necesidad más alta (cansancio = 1 - energía)
    return np.argmax(np.column_stack((X[:, 0], 1 - X[:, 1], X[:, 2])), axis=1)


def relu(v):
    return np.maximum(0, v)


def softmax(v):
    e = np.exp(v - np.max(v, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def entrenar(hidden=6, epochs=1200, lr=0.1, seed=42, verbose=False):
    np.random.seed(seed)
    X = np.random.rand(3000, 3)
    y = etiquetas(X)
    W1 = np.random.randn(3, hidden) * 0.5
    b1 = np.zeros((1, hidden))
    W2 = np.random.randn(hidden, 3) * 0.5
    b2 = np.zeros((1, 3))
    n = len(X)
    historial = []
    for epoch in range(epochs):
        z1 = X @ W1 + b1
        h = relu(z1)
        p = softmax(h @ W2 + b2)
        loss = -np.mean(np.log(p[np.arange(n), y] + 1e-10))
        d2 = p.copy()
        d2[np.arange(n), y] -= 1
        d2 /= n
        gW2 = h.T @ d2
        gb2 = d2.sum(axis=0, keepdims=True)
        d1 = (d2 @ W2.T) * (z1 > 0)
        gW1 = X.T @ d1
        gb1 = d1.sum(axis=0, keepdims=True)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
        historial.append(loss)
        if verbose and epoch % 200 == 0:
            print(f"  época {epoch:5d} | pérdida {loss:.4f}")

    def predecir(data):
        return softmax(relu(data @ W1 + b1) @ W2 + b2)

    test = np.random.rand(1000, 3)
    acc = np.mean(np.argmax(predecir(test), axis=1) == etiquetas(test))
    return predecir, historial, acc, (W1, b1, W2, b2)


if __name__ == "__main__":
    predecir, hist, acc, params = entrenar()
    print(f"BASE (6 neuronas, 1200 épocas, lr=0.1): pérdida final {hist[-1]:.4f} | precisión {acc:.2%}")

    # 1) Dos necesidades fijas, variar la tercera
    print("\n=== 1) Variar una necesidad (las otras fijas) ===")
    for nombre, idx, fijas in [
        ("Hambre", 0, (None, 0.7, 0.3)),   # energía=0.7 (cansancio .3), aburrimiento=.3
        ("Energía", 1, (0.5, None, 0.4)),  # hambre=.5, aburrimiento=.4
        ("Aburrimiento", 2, (0.4, 0.6, None)),  # hambre=.4, energía=.6 (cansancio .4)
    ]:
        prev = None
        print(f"-- varía {nombre}, fijas {[f for f in fijas if f is not None]}")
        for v in np.round(np.arange(0, 1.0001, 0.05), 2):
            x = list(fijas)
            x[idx] = v
            p = predecir(np.array([x]))[0]
            a = ACCIONES[int(np.argmax(p))]
            if a != prev:
                print(f"   {nombre}={v:.2f} -> {a}  p={np.round(p,3)}")
                prev = a

    # 2) Prioridades casi iguales
    print("\n=== 2) Prioridades casi iguales (hambre, energía, aburrimiento) ===")
    casos = [
        (0.50, 0.50, 0.50, "las tres exactamente iguales (cansancio=.5)"),
        (0.51, 0.50, 0.50, "hambre .51 vs resto .50"),
        (0.55, 0.50, 0.50, "hambre .55 vs resto .50"),
        (0.60, 0.50, 0.50, "hambre .60 vs resto .50"),
        (0.80, 0.80, 0.20, "hambre .80 vs cansancio .20 vs abur .20"),
        (0.80, 0.20, 0.20, "hambre .80 = cansancio .80 (energía .2), abur .2"),
        (0.90, 0.10, 0.10, "hambre .9, cansancio .9, abur .1 (empate)"),
        (0.95, 0.10, 0.10, "hambre .95, cansancio .9, abur .1"),
        (0.90, 0.10, 0.90, "hambre .9, cansancio .9, abur .9 (triple empate)"),
        (0.95, 0.90, 0.99, "caso extremo: todo muy alto"),
        (0.10, 0.90, 0.10, "todo bajo (estado NEUTRAL en la app)"),
    ]
    for h, e, a, desc in casos:
        p = predecir(np.array([[h, e, a]]))[0]
        print(f"   [{h:.2f},{e:.2f},{a:.2f}] {desc}\n      softmax={np.round(p,3)} -> {ACCIONES[int(np.argmax(p))]} "
              f"(política real: {ACCIONES[int(etiquetas(np.array([[h,e,a]]))[0])]})")

    # 3) Tamaño de la capa oculta
    print("\n=== 3) Capa oculta ===")
    print("neuronas | pérdida final | precisión test | semillas (media±desv precisión)")
    for hdn in (3, 6, 10, 20):
        _, h_, a_, _ = entrenar(hidden=hdn)
        accs = [entrenar(hidden=hdn, seed=s)[2] for s in range(10)]
        losses = [entrenar(hidden=hdn, seed=s)[1][-1] for s in range(10)]
        print(f"{hdn:8d} | {h_[-1]:.4f}        | {a_:.2%}        | acc {np.mean(accs):.2%}±{np.std(accs):.2%}, "
              f"pérdida {np.mean(losses):.4f}±{np.std(losses):.4f}, min acc {np.min(accs):.2%}")

    # 4) Épocas y tasa de aprendizaje
    print("\n=== 4) Épocas ===")
    for ep in (50, 200, 600, 1200, 3000):
        _, h_, a_, _ = entrenar(epochs=ep)
        print(f"épocas={ep:5d} | pérdida {h_[-1]:.4f} | precisión {a_:.2%}")
    print("\n=== 4b) Tasa de aprendizaje (1200 épocas) ===")
    for lr in (0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0):
        with np.errstate(all="ignore"):
            _, h_, a_, _ = entrenar(lr=lr)
        h_ = np.array(h_)
        saltos = int(np.sum(np.diff(h_) > 1e-3))
        print(f"lr={lr:6} | pérdida final {h_[-1]:.4f} | mín {np.nanmin(h_):.4f} | subidas de pérdida: {saltos} | precisión {a_:.2%}")

    # 5) Precisión engañosa: desbalance
    print("\n=== 5) Precisión alta puede engañar ===")
    np.random.seed(0)
    test = np.random.rand(100000, 3)
    y_t = etiquetas(test)
    print("Distribución de clases en la política sintética:",
          {ACCIONES[i]: f"{np.mean(y_t==i):.1%}" for i in range(3)})
    pred = np.argmax(predecir(test), axis=1)
    print(f"Precisión global (100k ejemplos): {np.mean(pred==y_t):.2%}")
    for i in range(3):
        m = y_t == i
        print(f"   recall {ACCIONES[i]}: {np.mean(pred[m]==i):.2%}")
    # Errores: ¿dónde se concentran?
    mal = pred != y_t
    srt = np.sort(np.column_stack((test[:, 0], 1 - test[:, 1], test[:, 2])), axis=1)
    margen = srt[:, 2] - srt[:, 1]
    print(f"Margen medio (1ª-2ª necesidad) en aciertos: {margen[~mal].mean():.3f} | en errores: {margen[mal].mean():.3f}")
    print(f"Errores con margen < 0.05: {np.mean(margen[mal] < 0.05):.1%} de los errores")

    # Modelo tramposo en datos desbalanceados (ejemplo hipotético)
    print("\nEjemplo hipotético: 95% de los datos son 'neutral'")
    y_h = np.array([0] * 950 + [1] * 50)
    print(f"   Un modelo que siempre dice 'neutral' acierta {np.mean(y_h==0):.0%} y nunca detecta el caso raro (recall 0%).")

    # Parámetros
    W1, b1, W2, b2 = params
    print("\n=== Parámetros de la red base ===")
    print("W1 (3x6):\n", np.round(W1, 2))
    print("b1:", np.round(b1, 2))
    print("W2 (6x3):\n", np.round(W2, 2))
    print("b2:", np.round(b2, 2))
    print("Total de parámetros:", W1.size + b1.size + W2.size + b2.size)
