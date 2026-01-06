import cv2
import socket
import base64
import json
import time
import numpy as np

# ==========================
# THIẾT LẬP THAM SỐ
# ==========================
class StreamingParams:
    """Các tham số cần thiết cho việc truyền video"""
    # Thông tin kết nối mạng
    SERVER_IP = "localhost"     # Địa chỉ IP của máy chủ
    SERVER_PORT = 6100          # Cổng kết nối
    
    # Thiết lập webcam
    IMG_WIDTH = 640             # Độ rộng hình ảnh
    IMG_HEIGHT = 480            # Độ cao hình ảnh
    
    # Tham số truyền dữ liệu
    SEND_DELAY = 2              # Khoảng cách giữa mỗi lần gửi (s)
    ENCODE_QUALITY = 80         # Độ nét của ảnh nén (0-100)

# ==========================
# KHỞI TẠO SOCKET
# ==========================
def setup_server_socket():
    """
    Tạo socket server và đợi client kết nối vào.
    
    Returns:
        client_conn: Kết nối socket với client
    """
    ip_addr = StreamingParams.SERVER_IP
    port_num = StreamingParams.SERVER_PORT
    
    # Khởi tạo socket kiểu TCP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Gán địa chỉ và bắt đầu lắng nghe
    sock.bind((ip_addr, port_num))
    sock.listen(1)
    
    print(f"[SERVER] Chờ kết nối tại {ip_addr}:{port_num}...")
    
    # Đợi và chấp nhận kết nối
    client_conn, client_addr = sock.accept()
    print(f"[SERVER] Client đã kết nối: {client_addr}")
    
    return client_conn, sock

# ==========================
# XỬ LÝ HÌNH ẢNH
# ==========================
def encode_image_to_text(img):
    """
    Biến đổi hình ảnh thành chuỗi ký tự để truyền đi.
    
    Các bước thực hiện:
    1. Nén hình ảnh sang định dạng JPEG
    2. Mã hóa dữ liệu nhị phân thành base64
    3. Trả về chuỗi text có thể truyền qua mạng
    
    Args:
        img: Mảng numpy chứa dữ liệu hình ảnh (định dạng BGR)
    
    Returns:
        str: Chuỗi base64 đại diện cho hình ảnh
    """
    # Nén ảnh sang JPEG
    is_ok, compressed = cv2.imencode(
        ".jpg", img,
        [int(cv2.IMWRITE_JPEG_QUALITY), StreamingParams.ENCODE_QUALITY]
    )
    
    if not is_ok:
        return None
    
    # Chuyển sang base64 để dễ truyền qua socket
    text_data = base64.b64encode(compressed.tobytes()).decode("utf-8")
    
    return text_data

def transmit_image(conn, img_data, img_idx):
    """
    Truyền một hình ảnh qua kết nối socket.
    
    Args:
        conn: Đối tượng socket đã kết nối
        img_data: Dữ liệu hình ảnh đã mã hóa
        img_idx: Số thứ tự của hình ảnh
    """
    # Đóng gói thông tin thành dictionary
    packet = {
        "frame_id": img_idx,
        "data": img_data,
        "timestamp": time.time(),
        "width": StreamingParams.IMG_WIDTH,
        "height": StreamingParams.IMG_HEIGHT
    }
    
    # Chuyển thành JSON và thêm ký tự kết thúc
    msg = json.dumps(packet) + "\n"
    
    # Gửi dữ liệu
    conn.sendall(msg.encode("utf-8"))

# ==========================
# CHƯƠNG TRÌNH CHÍNH
# ==========================
def run():
    """
    Hàm điều khiển chính của chương trình.
    
    Luồng hoạt động:
    1. Thiết lập socket và chờ client
    2. Khởi động webcam
    3. Vòng lặp: đọc ảnh -> mã hóa -> gửi đi
    """
    print("=" * 60)
    print("          VIDEO STREAMING SERVER")
    print("=" * 60)
    
    # Thiết lập kết nối socket
    print("\n[SERVER] Đang thiết lập socket...")
    client_socket, main_socket = setup_server_socket()
    
    # Khởi động webcam
    print("[SERVER] Đang khởi động webcam...")
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("[ERROR] Webcam không hoạt động!")
        print("[INFO] Sử dụng chế độ mô phỏng...")
        fake_mode = True
    else:
        fake_mode = False
        # Thiết lập thông số webcam
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, StreamingParams.IMG_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, StreamingParams.IMG_HEIGHT)
        print("[SERVER] Webcam hoạt động bình thường!")
    
    print(f"\n[SERVER] Bắt đầu truyền với delay = {StreamingParams.SEND_DELAY}s")
    print("[SERVER] Nhấn Ctrl+C để thoát...\n")
    
    img_count = 0
    t_start = time.time()
    
    try:
        while True:
            if fake_mode:
                # Tạo hình ảnh mô phỏng
                img = np.zeros((StreamingParams.IMG_HEIGHT, StreamingParams.IMG_WIDTH, 3), dtype=np.uint8)
                color_val = int((time.time() * 50) % 255)
                img[:, :] = [color_val, 100, 200 - color_val // 2]
                # Hiển thị số thứ tự
                cv2.putText(img, f"Frame #{img_count}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # Chụp hình từ webcam
                success, img = camera.read()
                
                if not success:
                    print("[WARN] Không thể chụp hình")
                    continue
                
                # Lật ngang hình ảnh
                img = cv2.flip(img, 1)
                
                # Thay đổi kích thước
                img = cv2.resize(img, (StreamingParams.IMG_WIDTH, StreamingParams.IMG_HEIGHT))
            
            # Mã hóa hình ảnh thành text
            encoded_data = encode_image_to_text(img)
            
            if encoded_data is None:
                print("[WARN] Lỗi khi mã hóa hình ảnh")
                continue
            
            # Gửi hình ảnh qua socket
            try:
                transmit_image(client_socket, encoded_data, img_count)
                img_count += 1
                
                # In thông tin
                t_elapsed = time.time() - t_start
                print(f"[SERVER] Gửi thành công #{img_count} | "
                      f"Thời gian: {t_elapsed:.1f}s | "
                      f"Kích thước: {len(encoded_data)} bytes")
                
            except BrokenPipeError:
                print("[ERROR] Mất kết nối với client!")
                break
            except Exception as err:
                print(f"[ERROR] Lỗi truyền dữ liệu: {err}")
                break
            
            # Delay trước khi gửi tiếp
            time.sleep(StreamingParams.SEND_DELAY)
    
    except KeyboardInterrupt:
        print("\n[SERVER] Đang tắt server...")
    
    finally:
        # Dọn dẹp tài nguyên
        if not fake_mode:
            camera.release()
        client_socket.close()
        main_socket.close()
        print("[SERVER] Đã giải phóng tài nguyên!")
        print(f"[SERVER] Tổng số hình đã gửi: {img_count}")


if __name__ == "__main__":
    run()
