#!/usr/bin/env python3
"""
Configuration Loader for MMS Benchmark
配置加载器 - 统一管理项目配置

Features:
- 自动加载YAML配置文件
- 支持环境变量覆盖
- 路径自动解析（相对路径转绝对路径）
- 单例模式（全局唯一配置实例）

Usage:
    from utils.config_loader import get_config

    # 获取配置实例
    config = get_config()

    # 访问配置
    videos_dir = config.get('videoxum.videos_dir')
    h5_file = config.get('videoxum.h5_file')
    workers = config.get('processing.default_workers', default=4)

    # 获取路径（自动转换为Path对象）
    videos_path = config.get_path('videoxum.videos_dir')

    # 检查环境
    if config.is_local():
        print("Running in local environment")
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional, Union
from multiprocessing import cpu_count


class Config:
    """配置管理类"""

    _instance = None  # 单例实例

    def __new__(cls):
        """单例模式：确保全局只有一个Config实例"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化配置"""
        if self._initialized:
            return

        # 查找项目根目录
        self.project_root = self._find_project_root()

        # 加载配置文件
        self.config_data = self._load_config()

        # 环境信息
        self.env_name = self.config_data.get('environment', {}).get('name', 'unknown')

        self._initialized = True

    def _find_project_root(self) -> Path:
        """查找项目根目录（包含configs目录的目录）"""
        current = Path(__file__).resolve().parent

        # 向上查找，直到找到configs目录
        while current != current.parent:
            if (current / 'configs').exists():
                return current
            current = current.parent

        # 如果没找到，使用当前文件的上两级目录
        return Path(__file__).resolve().parent.parent

    def _load_config(self) -> dict:
        """加载配置文件"""
        # 1. 从环境变量获取配置文件路径
        config_file = os.environ.get('MMS_CONFIG_FILE')

        # 2. 如果没有环境变量，尝试使用软链接
        if not config_file:
            config_file = 'configs/config.yaml'

        # 3. 构建完整路径
        config_path = self.project_root / config_file

        # 4. 如果配置文件不存在，尝试使用默认的local配置
        if not config_path.exists():
            print(f"Warning: Config file not found: {config_path}")
            print("Trying to use configs/config.local.yaml")
            config_path = self.project_root / 'configs' / 'config.local.yaml'

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Please run: ./switch_env.sh local  or  ./switch_env.sh server"
            )

        # 5. 加载YAML文件
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的嵌套键）

        Args:
            key: 配置键，支持嵌套（如 'videoxum.videos_dir'）
            default: 默认值

        Returns:
            配置值

        Examples:
            >>> config = get_config()
            >>> config.get('videoxum.videos_dir')
            '/path/to/videos'
            >>> config.get('processing.default_workers', default=4)
            8
        """
        keys = key.split('.')
        value = self.config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        # 特殊处理：如果是0（表示自动检测CPU核心数）
        if key.endswith('workers') and value == 0:
            return cpu_count()

        return value

    def get_path(self, key: str, default: Optional[str] = None) -> Path:
        """
        获取路径配置（自动转换为Path对象，并处理相对路径）

        Args:
            key: 配置键
            default: 默认路径

        Returns:
            Path对象（绝对路径）

        Examples:
            >>> config = get_config()
            >>> config.get_path('videoxum.videos_dir')
            PosixPath('/Users/moki/MMS_Benchmark/datasets_preprocessed/data/dataset/VideoXum/videos')
        """
        path_str = self.get(key, default=default)

        if path_str is None:
            return None

        path = Path(path_str)

        # 如果是相对路径，转换为相对于项目根目录的绝对路径
        if not path.is_absolute():
            path = self.project_root / path

        return path

    def is_local(self) -> bool:
        """判断是否是本地环境"""
        return self.env_name == 'local'

    def is_server(self) -> bool:
        """判断是否是服务器环境"""
        return self.env_name == 'server'

    def get_dataset_paths(self, dataset_name: str) -> dict:
        """
        获取指定数据集的所有路径（纯配置读取，无回退逻辑）

        所有路径必须在配置文件中显式定义：
        - 每个数据集需要定义: h5_file, videos_dir, frames_dir, frame_descriptions_dir, shot_descriptions_dir
        - 公共路径在 paths 下定义: processed_dir, statistics_dir, checkpoints_dir

        Args:
            dataset_name: 数据集名称（如 'summe', 'tvsum', 'videoxum'）

        Returns:
            dict: 包含以下键的路径字典（值为 Path 对象，未配置则为 None）
                - h5_file: H5标注文件路径
                - videos_dir: 视频文件目录
                - frames_dir: 抽帧输出目录
                - frame_descriptions_dir: 帧描述Excel中间文件目录
                - shot_descriptions_dir: 镜头描述Excel中间文件目录
                - processed_dir: JSON输出目录
                - statistics_dir: 统计文件目录
                - checkpoints_dir: checkpoint目录
        """
        ds = dataset_name.lower()

        return {
            'h5_file': self.get_path(f'{ds}.h5_file'),
            'videos_dir': self.get_path(f'{ds}.videos_dir'),
            'frames_dir': self.get_path(f'{ds}.frames_dir'),
            'frame_descriptions_dir': self.get_path(f'{ds}.frame_descriptions_dir'),
            'shot_descriptions_dir': self.get_path(f'{ds}.shot_descriptions_dir'),
            'processed_dir': self.get_path('paths.processed_dir'),
            'statistics_dir': self.get_path('paths.statistics_dir'),
            'checkpoints_dir': self.get_path('paths.checkpoints_dir'),
        }

    def detect_datasets(self) -> list:
        """
        从配置文件中获取可用的数据集列表

        检测逻辑：扫描配置文件中定义了 h5_file 的顶级键

        Returns:
            list: 数据集名称列表（小写）
        """
        datasets = []
        # 已知的数据集名称（配置文件中的顶级键）
        known_datasets = ['summe', 'tvsum', 'ovp', 'youtube', 'videoxum', 'mrhisum']

        for ds in known_datasets:
            # 检查配置中是否定义了该数据集的 h5_file
            if self.get(f'{ds}.h5_file'):
                datasets.append(ds)

        return datasets

    def get_all(self) -> dict:
        """获取所有配置"""
        return self.config_data.copy()

    def print_config(self):
        """打印当前配置（用于调试）"""
        print("\n" + "="*60)
        print(f"Environment: {self.env_name}")
        print(f"Project Root: {self.project_root}")
        print("="*60)
        print("\nKey Paths:")
        print(f"  VideoXum videos: {self.get_path('videoxum.videos_dir')}")
        print(f"  VideoXum frames: {self.get_path('videoxum.frames_dir')}")
        print(f"  VideoXum H5: {self.get_path('videoxum.h5_file')}")
        print(f"  Log directory: {self.get_path('paths.log_dir')}")
        print("\nProcessing Settings:")
        print(f"  Workers: {self.get('processing.default_workers')}")
        print(f"  Use FFmpeg: {self.get('processing.use_ffmpeg')}")
        print(f"  Delete after: {self.get('processing.delete_after_extraction')}")
        print("="*60 + "\n")


# 全局配置实例
_global_config = None


def get_config() -> Config:
    """
    获取全局配置实例（单例模式）

    Returns:
        Config实例

    Examples:
        >>> config = get_config()
        >>> videos_dir = config.get_path('videoxum.videos_dir')
    """
    global _global_config

    if _global_config is None:
        _global_config = Config()

    return _global_config


def reload_config():
    """重新加载配置（用于切换环境后）"""
    global _global_config
    _global_config = None
    return get_config()


# 便捷函数
def get_dataset_paths(dataset_name: str) -> dict:
    """获取指定数据集的所有路径"""
    config = get_config()
    return config.get_dataset_paths(dataset_name)


def get_videoxum_config() -> dict:
    """获取VideoXum数据集配置"""
    return get_dataset_paths('videoxum')


def get_summe_config() -> dict:
    """获取SumMe数据集配置"""
    return get_dataset_paths('summe')


def get_processing_config() -> dict:
    """获取处理参数配置"""
    config = get_config()
    return {
        'workers': config.get('processing.default_workers'),
        'use_ffmpeg': config.get('processing.use_ffmpeg'),
        'delete_after': config.get('processing.delete_after_extraction'),
        'watch_interval': config.get('processing.watch_interval'),
    }


if __name__ == '__main__':
    # 测试配置加载
    config = get_config()
    config.print_config()

    # 测试获取配置
    print("\nTesting get() method:")
    print(f"  Environment: {config.get('environment.name')}")
    print(f"  Workers: {config.get('processing.default_workers')}")
    print(f"  Use FFmpeg: {config.get('processing.use_ffmpeg')}")

    print("\nTesting get_path() method:")
    print(f"  Videos dir: {config.get_path('videoxum.videos_dir')}")
    print(f"  H5 file: {config.get_path('videoxum.h5_file')}")

    print("\nTesting environment check:")
    print(f"  Is local: {config.is_local()}")
    print(f"  Is server: {config.is_server()}")

    print("\nTesting convenience functions:")
    print(f"  VideoXum config: {get_videoxum_config()}")
    print(f"  Processing config: {get_processing_config()}")
