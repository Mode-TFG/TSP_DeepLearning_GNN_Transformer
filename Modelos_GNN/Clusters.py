import matplotlib.pyplot as plt

# Métricas exactas de la última ejecución limpia (sin artefactos JIT)
escenarios = ['20 nodos\n(Uniforme)', '50 nodos\n(Uniforme)', '100 nodos\n(Uniforme)', '20 nodos\n(Clústeres)']
gaps = [1.81, 4.69, 4.15, 0.61]

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
bars = ax.bar(escenarios, gaps, color='#1f77b4')

# Etiquetas de valor exacto sobre cada barra
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval}%", ha='center', va='bottom', fontsize=11)

ax.set_ylabel('Gap medio frente a OR-Tools (%)')
ax.set_title('Resiliencia topológica: Escalabilidad dimensional y distribuciones heterogéneas')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()

# Sobrescribe la imagen desactualizada
plt.savefig('figuras_tfg/grafica_generalizacion.png')
print("Gráfica regenerada y guardada con éxito.")