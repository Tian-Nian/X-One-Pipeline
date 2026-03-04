#!/usr/bin/env python3
import os
import subprocess
import sys

def get_camera_info():
    """获取当前所有 video 设备的 ID_SERIAL_SHORT"""
    cameras = []
    try:
        # 获取所有 video4linux 设备
        output = subprocess.check_output("ls /dev/video*", shell=True).decode().split()
        video_devices = [v for v in output if v.strip()]
        
        seen_serials = set()
        for dev in video_devices:
            try:
                # 只获取主设备节点（index 0）的信息
                info = subprocess.check_output(f"udevadm info --query=all --name={dev}", shell=True).decode()
                if "ID_USB_INTERFACE_NUM=00" not in info: # 过滤掉同一个摄像头的辅助流（通常序号不是00）
                    continue
                
                vendor_id = ""
                model_id = ""
                serial = ""
                
                for line in info.splitlines():
                    if line.startswith("E: ID_VENDOR_ID="):
                        vendor_id = line.split("=")[1].strip()
                    if line.startswith("E: ID_MODEL_ID="):
                        model_id = line.split("=")[1].strip()
                    if line.startswith("E: ID_SERIAL_SHORT="):
                        serial = line.split("=")[1].strip()
                
                if serial and serial not in seen_serials:
                    cameras.append({
                        'dev': dev,
                        'vendor': vendor_id,
                        'model': model_id,
                        'serial': serial
                    })
                    seen_serials.add(serial)
            except:
                continue
    except Exception as e:
        print(f"扫描设备失败: {e}")
    
    return cameras

def main():
    if os.getuid() != 0:
        print("错误: 请使用 sudo 运行此程序以生成 udev 规则！")
        print("用法: sudo python3 tools/set_camera_rules.py")
        sys.exit(1)

    print("正在扫描已连接的 USB 摄像头...")
    cameras = get_camera_info()
    
    if not cameras:
        print("未能识别到有效的 USB 摄像头。")
        return

    print(f"找到 {len(cameras)} 个摄像头设备:")
    for i, cam in enumerate(cameras):
        print(f"[{i}] 设备: {cam['dev']}, VendorID: {cam['vendor']}, ModelID: {cam['model']}, Serial: {cam['serial']}")

    mappings = {
        'head_camera': None,
        'left_wrist_camera': None,
        'right_wrist_camera': None
    }

    print("\n请按提示分配摄像头角色 (输入索引号):")
    for role in mappings.keys():
        while True:
            choice = input(f"请输入 {role} 的索引 (跳过请直接回车): ").strip()
            if not choice:
                break
            try:
                idx = int(choice)
                if 0 <= idx < len(cameras):
                    mappings[role] = cameras[idx]
                    break
                else:
                    print("无效的索引，请重试。")
            except ValueError:
                print("输入不是有效的数字。")

    # 生成 udev 规则
    rule_content = ""
    assigned = False
    for role in ['head_camera', 'left_wrist_camera', 'right_wrist_camera']:
        cam = mappings.get(role)
        if cam:
            line = (
                f'SUBSYSTEM=="video4linux", KERNEL=="video*", '
                f'ATTRS{{idVendor}}=="{cam["vendor"]}", ATTRS{{idProduct}}=="{cam["model"]}", '
                f'ATTRS{{serial}}=="{cam["serial"]}", ATTR{{index}}=="0", '
                f'SYMLINK+="{role}"\n'
            )
            rule_content += line
            assigned = True

    if not assigned:
        print("未进行任何分配。")
        return

    rule_path = "/etc/udev/rules.d/99-usb-cameras.rules"
    try:
        with open(rule_path, "w") as f:
            f.write(rule_content)
        
        print(f"\n规则已写入 {rule_path}")
        print("正在重载 udev 规则...")
        subprocess.run(["udevadm", "control", "--reload-rules"], check=True)
        subprocess.run(["udevadm", "trigger"], check=True)
        print("\n成功！您现在可以使用以下路径访问摄像头:")
        for role, cam in mappings.items():
            if cam:
                print(f"  - /dev/{role} -> (Vendor:{cam['vendor']}, Serial:{cam['serial']})")
        print("\n注意: 如果符号链接未立即生效，请尝试重新拔插摄像头。")
    except Exception as e:
        print(f"写入规则失败: {e}")

if __name__ == "__main__":
    main()
