#!/usr/bin/env python3
import os
import subprocess
import sys

def get_device_info():
    """获取当前所有 USB 设备的属性"""
    devices = []
    try:
        # 扫描 block, char, video4linux, tty 等各种设备
        # 我们通过 udevadm 直接枚举所有 USB 相关的父设备
        output = subprocess.check_output("find -L /sys/bus/usb/devices/ -maxdepth 2 -name \"idVendor\"", shell=True).decode().split()
        
        seen_serials = set()
        for vendor_file in output:
            try:
                # 获取设备路径
                dev_path = os.path.dirname(vendor_file)
                
                # 获取属性
                with open(os.path.join(dev_path, "idVendor"), 'r') as f:
                    vendor_id = f.read().strip()
                with open(os.path.join(dev_path, "idProduct"), 'r') as f:
                    model_id = f.read().strip()
                
                serial = ""
                serial_path = os.path.join(dev_path, "serial")
                if os.path.exists(serial_path):
                    with open(serial_path, 'r') as f:
                        serial = f.read().strip()
                
                # 寻找对应的 /dev/ 节点
                # 这有点复杂，因为一个 USB 设备可能有多个接口和端点
                # 我们尝试通过 udevadm 找该路径下的所有 devname
                dev_list = []
                try:
                    udev_info = subprocess.check_output(f"udevadm info --export-db", shell=True).decode()
                    # 这里通过解析 udev 数据库来找到属于该 USB 拓扑路径的所有设备节点
                    # 简化处理：直接看该路径下的子目录是否有 video*, tty*, hidraw* 等
                    for root, dirs, files in os.walk(dev_path):
                        for d in dirs:
                            # 典型的设备节点模式
                            if any(d.startswith(prefix) for prefix in ['video', 'tty', 'hidraw', 'ttyACM', 'ttyUSB', 'i2c-']):
                                dev_list.append(d)
                except:
                    pass

                if serial and serial not in seen_serials:
                    devices.append({
                        'dev_list': list(set(dev_list)),
                        'vendor': vendor_id,
                        'model': model_id,
                        'serial': serial,
                        'path': dev_path
                    })
                    seen_serials.add(serial)
            except:
                continue
    except Exception as e:
        print(f"扫描设备失败: {e}")
    
    return devices

def main():
    if os.getuid() != 0:
        print("错误: 请使用 sudo 运行此程序以生成 udev 规则！")
        print("用法: sudo python3 tools/set_usb_rules.py")
        sys.exit(1)

    print("正在扫描已连接的 USB 设备...")
    devices = get_device_info()
    
    if not devices:
        print("未能识别到有效的 USB 设备。")
        return

    print(f"找到 {len(devices)} 个设备:")
    for i, dev in enumerate(devices):
        dev_nodes = ", ".join(dev['dev_list']) or "不支持识别"
        print(f"[{i}] 设备节点: {dev_nodes}, VendorID: {dev['vendor']}, ModelID: {dev['model']}, Serial: {dev['serial']}")

    mappings = [] # 改为列表以支持任意数量的分配

    print("\n请按提示分配设备角色 (输入索引号 或 直接输入 /dev/ 路径，例如 /dev/hidraw7):")
    while True:
        choice = input(f"请输入要命名的设备索引或路径 (完成按回车): ").strip()
        if not choice:
            break
        
        target_device = None
        
        # 尝试作为 /dev/ 路径处理
        if choice.startswith("/dev/"):
            dev_node = choice[5:]
            # 查找哪个设备包含这个节点
            for d in devices:
                if dev_node in d['dev_list']:
                    target_device = d
                    break
            if not target_device:
                print(f"错误: 未在扫描结果中找到使用 {choice} 的 USB 设备。")
                continue
        else:
            # 尝试作为索引处理
            try:
                idx = int(choice)
                if 0 <= idx < len(devices):
                    target_device = devices[idx]
                else:
                    print("无效的索引，请重试。")
                    continue
            except ValueError:
                print("输入不是有效的数字或 /dev/ 路径。")
                continue

        if target_device:
            target_name = input(f"请输入为该设备设置的别名 (例如 pedal): ").strip()
            if target_name:
                # 提取内核节点的前缀 (例如 hidraw, video, ttyUSB)
                kernel_pattern = "*"
                if choice.startswith("/dev/"):
                    import re
                    match = re.search(r'([a-zA-Z]+)\d+', choice[5:])
                    if match:
                        kernel_pattern = match.group(1) + "*"
                elif target_device['dev_list']:
                    # 从索引选定时，尝试从其第一个节点提取前缀
                    import re
                    match = re.search(r'([a-zA-Z]+)\d+', target_device['dev_list'][0])
                    if match:
                        kernel_pattern = match.group(1) + "*"
                
                mappings.append({
                    'device': target_device,
                    'symlink': target_name,
                    'kernel': kernel_pattern
                })
            else:
                print("名称不能为空，已跳过。")

    # 生成 udev 规则
    rule_content = ""
    assigned = False
    for map_item in mappings:
        dev = map_item['device']
        symlink = map_item['symlink']
        kernel = map_item.get('kernel', '*')
        
        # 动态匹配内核节点类型 (如 KERNEL=="video*" 或 KERNEL=="hidraw*")
        line = (
            f'KERNEL=="{kernel}", SUBSYSTEMS=="usb", ATTRS{{idVendor}}=="{dev["vendor"]}", '
            f'ATTRS{{idProduct}}=="{dev["model"]}", ATTRS{{serial}}=="{dev["serial"]}", '
            f'SYMLINK+="{symlink}"\n'
        )
        rule_content += line
        assigned = True

    if not assigned:
        print("未进行任何分配。")
        return

    rule_path = "/etc/udev/rules.d/99-usb-custom.rules"
    try:
        with open(rule_path, "w") as f:
            f.write(rule_content)
        
        print(f"\n规则已写入 {rule_path}")
        print("正在重载 udev 规则...")
        subprocess.run(["udevadm", "control", "--reload-rules"], check=True)
        subprocess.run(["udevadm", "trigger"], check=True)
        print("\n成功！您现在可以使用定义的别名访问设备:")
        for map_item in mappings:
            dev = map_item['device']
            symlink = map_item['symlink']
            print(f"  - /dev/{symlink} -> (Vendor:{dev['vendor']}, Serial:{dev['serial']})")
        print("\n注意: 如果符号链接未立即生效，请尝试重新拔插设备。")
    except Exception as e:
        print(f"写入规则失败: {e}")

if __name__ == "__main__":
    main()
