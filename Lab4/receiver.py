import json
import base64
import numpy as np
import cv2
from datetime import datetime
import warnings
import sys
import os

lab4_path = os.path.join(os.path.dirname(__file__), "Lab4")
if lab4_path not in sys.path:
    sys.path.insert(0, lab4_path)

from background_remover import remove_background

warnings.filterwarnings("ignore")

# Thiết lập tham số cho Spark trước khi load module
os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.ui.showConsoleProgress=false pyspark-shell'

# Cấu hình đường dẫn Python interpreter cho môi trường Spark
py_interpreter = sys.executable
os.environ['PYSPARK_PYTHON'] = py_interpreter
os.environ['PYSPARK_DRIVER_PYTHON'] = py_interpreter

from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext

# Lớp chứa các thông số cấu hình
class StreamConfig:
    """Thiết lập các tham số kết nối và xử lý stream"""
    HOST_ADDR = "localhost"
    PORT_NUM = 6100
    
    SPARK_APP = "ImageBackgroundRemover"
    SPARK_MASTER = "local[*]"
    INTERVAL_SEC = 2

# ==========================
# HÀM CHUYỂN ĐỔI DỮ LIỆU
# ==========================
def string_to_frame(base64_string):
    """
    Chuyển đổi chuỗi base64 thành frame (ma trận numpy).
    Đây là quá trình giải mã ngược lại từ văn bản về số.
    """
    try:
        # Giải mã base64 string thành bytes
        img_bytes = base64.b64decode(base64_string)
        
        # Chuyển bytes thành numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode thành frame (ma trận BGR)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return frame
    except Exception as e:
        print(f"[ERROR] Lỗi giải mã frame: {e}")
        return None

# ==========================
# HÀM PARSE JSON (CHẠY TRONG SPARK WORKER)
# ==========================
def parse_json_line(line):
    """
    Parse JSON từ một dòng dữ liệu.
    Hàm này chạy trong Spark worker - chỉ làm việc nhẹ (parse JSON).
    KHÔNG import thư viện nặng như MediaPipe ở đây.
    """
    try:
        data = json.loads(line)
        return {
            "success": True,
            "frame_id": data.get("frame_id", 0),
            "base64_data": data.get("data", ""),
            "width": data.get("width", 640),
            "height": data.get("height", 480)
        }
    except Exception as e:
        return {"success": False, "frame_id": -1, "error": str(e)}

# ==========================
# HÀM XỬ LÝ FRAME (CHẠY TRONG DRIVER)
# ==========================
def process_frame_in_driver(parsed_data):
    """
    Xử lý frame trong driver (sau khi collect từ RDD).
    Gọi remove_background ở đây vì MediaPipe không thể serialize.
    """
    if not parsed_data["success"]:
        return {
            "success": False,
            "frame_id": parsed_data.get("frame_id", -1),
            "message": parsed_data.get("error", "Unknown error")
        }
    
    frame_id = parsed_data["frame_id"]
    base64_data = parsed_data["base64_data"]
    
    # Chuyển base64 thành frame
    frame = string_to_frame(base64_data)
    
    if frame is None:
        return {
            "success": False,
            "frame_id": frame_id,
            "message": "Lỗi giải mã frame"
        }
    
    # Xử lý xóa phông nền (trong driver)
    processed_frame = remove_background(frame)
    
    # Lưu frame đã xử lý
    output_dir = os.path.join(os.path.dirname(__file__), "output_frames")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
    cv2.imwrite(output_path, processed_frame)
    
    return {
        "success": True,
        "frame_id": frame_id,
        "original_shape": frame.shape,
        "processed_shape": processed_frame.shape,
        "output_path": output_path
    }

# ==========================
# HÀM XỬ LÝ RDD
# ==========================
def process_rdd(time, rdd):

    print("\n" + "=" * 50)
    print(f"[RECEIVER] Batch time: {time}")
    print("=" * 50)
    
    try:
        # Đếm số records trong RDD
        count = rdd.count()
        
        if count == 0:
            print("[RECEIVER] Batch trống - không có dữ liệu")
            return
        
        print(f"[RECEIVER] Số frame trong batch: {count}")
        
        # BƯỚC 1: Sử dụng RDD.map() để parse JSON (RDD operation - trong ngữ cảnh Spark)
        # Đây là xử lý phân tán, đúng với yêu cầu dùng RDD
        parsed_rdd = rdd.map(parse_json_line)
        
        # BƯỚC 2: Collect kết quả về driver
        # Bỏ qua lỗi Python 3.13 socket nếu có
        try:
            results = parsed_rdd.collect()
        except Exception as e:
            if "socket" in str(e).lower() or "10038" in str(e):
                print("[RECEIVER] Bỏ qua lỗi socket Python 3.13")
                return
            raise
        
        # BƯỚC 3: Xử lý từng frame đã parse
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        success_count = 0
        for parsed_data in results:
            result = process_frame_in_driver(parsed_data)
            if result["success"]:
                success_count += 1
                print(f"[RECEIVER] Frame #{result['frame_id']}: {result['original_shape']} -> Saved")
            else:
                print(f"[RECEIVER] Frame #{result['frame_id']}: {result['message']}")
        
        print(f"[RECEIVER] Hoàn thành batch! Đã xử lý {success_count}/{count} frame(s)")
        
    except Exception as e:
        # Bỏ qua lỗi socket
        if "socket" not in str(e).lower() and "10038" not in str(e):
            print(f"[RECEIVER] Lỗi xử lý batch: {e}")

# ==========================
# MAIN - SPARK STREAMING
# ==========================
def main():
    """
    Hàm chính để chạy Receiver module với Spark Streaming.
    """
    print("=" * 60)
    print("     RECEIVER MODULE - SPARK STREAMING SERVER")
    print("=" * 60)
    print(f"\n[INFO] Python executable: {sys.executable}")
    
    # ==========================================
    # BƯỚC 1: Khởi tạo SparkContext
    # ==========================================
    print("\n[RECEIVER] Đang khởi tạo SparkContext...")
    
    # Tạo SparkConf với các cấu hình
    conf = SparkConf()
    conf.setAppName(StreamConfig.SPARK_APP)
    conf.setMaster(StreamConfig.SPARK_MASTER)
    
    # Tạo SparkContext
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")  # Chỉ hiển thị lỗi
    
    print(f"[RECEIVER] SparkContext đã khởi tạo!")
    print(f"           - App Name: {StreamConfig.SPARK_APP}")
    print(f"           - Master: {StreamConfig.SPARK_MASTER}")
    
    # ==========================================
    # BƯỚC 2: Khởi tạo StreamingContext
    # ==========================================
    print("\n[RECEIVER] Đang khởi tạo StreamingContext...")
    
    # StreamingContext với batch interval
    ssc = StreamingContext(sc, StreamConfig.INTERVAL_SEC)
    
    print(f"[RECEIVER] StreamingContext đã khởi tạo!")
    print(f"           - Batch Interval: {StreamConfig.INTERVAL_SEC} giây")
    
    # ==========================================
    # BƯỚC 3: Kết nối Socket Text Stream
    # ==========================================
    print("\n[RECEIVER] Đang kết nối đến Sender...")
    print(f"           - Host: {StreamConfig.HOST_ADDR}")
    print(f"           - Port: {StreamConfig.PORT_NUM}")
    
    # Sử dụng socketTextStream để nhận dữ liệu dạng văn bản
    lines = ssc.socketTextStream(
        StreamConfig.HOST_ADDR, 
        StreamConfig.PORT_NUM
    )
    
    # ==========================================
    # BƯỚC 4: Xử lý DStream với Spark
    # ==========================================
    # Áp dụng foreachRDD để xử lý từng batch RDD
    lines.foreachRDD(process_rdd)
    
    # ==========================================
    # BƯỚC 5: Bắt đầu Streaming
    # ==========================================
    print("\n" + "=" * 60)
    print("[RECEIVER] Bắt đầu Spark Streaming...")
    print("[RECEIVER] Đang chờ dữ liệu từ Sender...")
    print("[RECEIVER] Nhấn Ctrl+C để dừng...")
    print("=" * 60 + "\n")
    
    # Bắt đầu streaming
    ssc.start()
    
    try:
        # Chờ cho đến khi streaming kết thúc
        ssc.awaitTermination()
    except KeyboardInterrupt:
        print("\n[RECEIVER] Đang dừng streaming...")
    finally:
        ssc.stop(stopSparkContext=True, stopGraceFully=True)
        print("[RECEIVER] Spark Streaming đã dừng!")


if __name__ == "__main__":
    main()
