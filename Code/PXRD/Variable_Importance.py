import matplotlib.pyplot as plt


# Visualize Importance (Graph)
plt.figure(figsize=(10,6))
plt.barh(Importance_FPF_DF['Feature'][::-1], Importance_FPF_DF['Importance'][::-1], color='teal')
plt.xlabel('Relative Importance')
plt.ylabel('Features (Experimental conditions & 2-Theta Angles)')
plt.title('Feature Importance for FPF(%) Prediction')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
