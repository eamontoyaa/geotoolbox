import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def fit_distributions(X, dist_names=["lognorm", "gamma", "weibull_min"]):
    # Reemplazar valores cero con un número pequeño si es necesario
    X = np.where(np.isclose(X, 0.0, atol=1e-12), 1e-3, X)

    # Ajustar las distribuciones y almacenar los parámetros
    fitted_params = {}
    for dist_name in dist_names:
        try:
            # Obtener la clase de distribución de scipy.stats
            dist_class = getattr(sp.stats, dist_name)
            # Ajustar la distribución con floc=0 para evitar valores negativos de loc
            fitted_params[dist_name] = dist_class.fit(X, floc=0)
        except AttributeError as e:
            raise ValueError(
                f"La distribución '{dist_name}' no es válida en scipy.stats."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Error al ajustar la distribución '{dist_name}': {e}")

    return fitted_params



def plot_fitted_distributions(X, fitted_params):
    # Generar puntos para las curvas ajustadas
    x_plot = np.linspace(X.min(), X.max(), 1000)

    # Calcular la función de distribución acumulada empírica (CDF)
    sorted_data = np.sort(X)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Crear figura y ejes
    fig, axs = plt.subplots(
        ncols=1, nrows=2, figsize=(6, 6), sharex=True, layout="constrained"
    )

    # Graficar histograma
    axs[0].hist(
        X,
        bins="auto",
        density=True,
        facecolor="k",
        edgecolor="w",
        # linewidth=1,
        label="Histograma",
    )
    # Guarda limites del eje y
    y_min, y_max = axs[0].get_ylim()

    # Graficar CDF empírica
    empirical_cdf_line = axs[1].plot(
        sorted_data,
        empirical_cdf,
        label="Función de distribución\nacumulada empírica",
        marker="o",
        ms=7.5,
        ls="",
        mec="w",
        mfc="k",
    )

    # Graficar distribuciones ajustadas
    for i, (dist_name, params) in enumerate(fitted_params.items()):
        # Manejar el caso especial de Weibull
        if dist_name == "weibull":
            label = "Ajuste Weibull"
            dist_name = "weibull_min"
        else:
            label = f"Ajuste {dist_name.capitalize()}"

        try:
            dist_class = getattr(sp.stats, dist_name)
            dist = dist_class(*params)
        except AttributeError as e:
            raise ValueError(
                f"La distribución '{dist_name}' no es válida en scipy.stats."
            ) from e

        # PDF (Densidad de probabilidad)
        axs[0].plot(x_plot, dist.pdf(x_plot), c=f"C{i}", label=label, ls="--")
        # CDF (Función de distribución acumulada)
        axs[1].plot(x_plot, dist.cdf(x_plot), c=f"C{i}", label=label, ls="--")

    # Etiquetas de los ejes
    axs[0].set_ylabel("Densidad de probabilidad")
    axs[1].set_ylabel("Probabilidad acumulada")
    axs[1].set_xlabel("Distancia recorrida [m]")

    # Restablecer límites del eje y del histograma
    axs[0].set_ylim(y_min, y_max)

    # Configurar cuadrícula
    for ax in axs:
        ax.grid(ls="--")
        ax.spines[["left", "bottom"]].set_linewidth(1.5)
        ax.tick_params(width=1.5)

    # Crear una leyenda combinada
    handles, labels = axs[0].get_legend_handles_labels()
    # Agregar la línea de la CDF empírica a la leyenda
    handles += empirical_cdf_line
    labels += ["Función de distribución\nacumulada empírica"]

    # Ubicar la leyenda fuera de los gráficos
    fig.legend(handles, labels, loc="outside center right")

    return fig




def select_distribution(X, fitted_params, manual=None):
    X = np.where(np.isclose(X, 0.0, atol=1e-12), 1e-3, X)
    
    # Inicializar resultados de KS y contenedor para distribuciones ajustadas
    ks_stat = {}
    all_dists = {}

    print("Resultados de la prueba de Kolmogorov-Smirnov:")

    # Evaluar cada distribución ajustada
    for dist_name, params in fitted_params.items():
        try:
            # Obtener la clase de distribución de scipy.stats
            dist_class = getattr(sp.stats, dist_name)
            dist = dist_class(*params)

            # Calcular la estadística de KS
            ks_stat[dist_name] = sp.stats.kstest(X, dist.cdf)[0]

            # Almacenar la instancia de la distribución
            all_dists[dist_name] = dist

            # Imprimir el resultado de KS para la distribución
            print(f"{dist_name.capitalize()}: {ks_stat[dist_name]:.3f}")
        except AttributeError:
            raise ValueError(
                f"La distribución '{dist_name}' no es válida en scipy.stats."
            )
        except Exception as e:
            raise RuntimeError(f"Error al evaluar la distribución '{dist_name}': {e}")

    # Seleccionar la distribución
    if manual:
        selected_dist_name = manual.lower()
        if selected_dist_name not in all_dists:
            raise ValueError(
                f"La distribución seleccionada manualmente '{manual}' no está en las opciones ajustadas."
            )
    else:
        # Selección automática basada en la menor estadística de KS
        selected_dist_name = min(ks_stat, key=ks_stat.get)

    print(f"\nDistribución seleccionada → {selected_dist_name.capitalize()}")

    # Retornar la distribución seleccionada
    return all_dists[selected_dist_name]
