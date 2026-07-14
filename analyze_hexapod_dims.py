
import os

def parse_obj_file(file_path):
    """解析OBJ文件，返回顶点坐标的min/max和尺寸"""
    min_x, max_x = float('inf'), -float('inf')
    min_y, max_y = float('inf'), -float('inf')
    min_z, max_z = float('inf'), -float('inf')
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = map(float, parts[1:4])
                        min_x, max_x = min(min_x, x), max(max_x, x)
                        min_y, max_y = min(min_y, y), max(max_y, y)
                        min_z, max_z = min(min_z, z), max(max_z, z)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    if min_x == float('inf'):
        return None
    
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    
    return {
        'file': os.path.basename(file_path),
        'min': (min_x, min_y, min_z),
        'max': (max_x, max_y, max_z),
        'size': (size_x, size_y, size_z)
    }

def main():
    obj_dir = '/home/ubuntu/IsaacLab-LiBRHexapod/hexapod-assets/OBJ'
    
    print("="*80)
    print("六足机器人各部件尺寸分析")
    print("="*80)
    
    # 解析所有OBJ文件
    parts_info = []
    for filename in os.listdir(obj_dir):
        if filename.endswith('.obj'):
            file_path = os.path.join(obj_dir, filename)
            info = parse_obj_file(file_path)
            if info:
                parts_info.append(info)
                print(f"\n{filename}:")
                print(f"  尺寸 (x, y, z): {info['size'][0]:.4f}m x {info['size'][1]:.4f}m x {info['size'][2]:.4f}m")
                print(f"  范围 x: {info['min'][0]:.4f}m ~ {info['max'][0]:.4f}m")
                print(f"  范围 y: {info['min'][1]:.4f}m ~ {info['max'][1]:.4f}m")
                print(f"  范围 z: {info['min'][2]:.4f}m ~ {info['max'][2]:.4f}m")
    
    # 汇总主链路部件尺寸
    print("\n" + "="*80)
    print("主要部件尺寸汇总")
    print("="*80)
    
    main_body_parts = {'CenterLink.obj', 'FrontLink.obj', 'BackLink.obj'}
    total_length = 0.0
    max_width = 0.0
    max_height = 0.0
    
    for info in parts_info:
        if info['file'] in main_body_parts:
            length = info['size'][0]  # x方向通常是长度
            width = info['size'][1]   # y方向
            height = info['size'][2]  # z方向
            
            total_length += length
            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height
    
    print(f"\n主体长度（前Link + 中Link + 后Link）: {total_length:.4f}米")
    print(f"主体最大宽度: {max_width:.4f}米")
    print(f"主体最大高度: {max_height:.4f}米")
    
    # 估计整体尺寸（包括腿的展开）
    print("\n" + "="*80)
    print("机器人整体估计尺寸")
    print("="*80)
    print(f"体长（主体）: {total_length:.4f}米")
    print(f"体宽（含展开的腿，估计）: ~0.3米")
    print(f"体高（含站立的腿，估计）: ~0.15米")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
