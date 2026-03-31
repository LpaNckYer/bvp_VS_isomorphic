import logging
import sys
from datetime import datetime

def setup_logging(log_file='run.log', level=logging.INFO):
    """设置日志记录"""
    # 创建日志器
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger



# 使用示例
if __name__ == "__main__":
    # 设置日志
    logger = setup_logging('run.log')
    
    logger.info("程序开始运行")
    
    try:
        # 你的程序代码
        logger.info("开始处理数据...")
        
        for i in range(10):
            logger.debug(f"处理第 {i+1} 条记录")
            
            if i == 5:
                logger.warning("遇到特殊情况")
            
            # 模拟处理
            import time
            time.sleep(0.1)
        
        logger.info("数据处理完成")
        
    except Exception as e:
        logger.error(f"程序出错: {e}", exc_info=True)
    
    finally:
        logger.info("程序结束")