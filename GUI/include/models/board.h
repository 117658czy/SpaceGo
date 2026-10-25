// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：board.h
// 描述：棋盘模型
// 创建日期：2026-02-21
// 作者：SpaceGo Team
// ============================================================================

#pragma once
#include <vector>
#include "models/piece.h"

namespace SpaceGo {
namespace GUI {
namespace Models {

class Board {
public:
    int gridSize;
    std::vector<std::vector<Piece>> pieces;

    Board(int size = 19);
    ~Board();

    // 重置棋盘
    void Reset();

    // 设置棋盘大小
    void SetSize(int size);

    // 放置棋子
    bool PlacePiece(int x, int y, PieceColor color, int moveNumber = 0);

    // 移除棋子
    void RemovePiece(int x, int y);

    // 获取棋子
    Piece GetPiece(int x, int y) const;

    // 检查位置是否有效
    bool IsValidPosition(int x, int y) const;

    // 检查位置是否为空
    bool IsEmpty(int x, int y) const;

    // 获取棋子数量
    int GetPieceCount() const;
};

} // namespace Models
} // namespace GUI
} // namespace SpaceGo
