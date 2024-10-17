from datetime import datetime
import matplotlib.pyplot as plt

def save_img_with_timestamp(figname, prefix="", format="png"):
  format = format if format in ['jpg', 'png', 'jpeg', 'svg', 'pdf', 'eps'] else 'png'
  plt.savefig(f"{prefix}{figname}_{datetime.now().strftime('%m_%d_%H_%M_%S')}.{prefix}")