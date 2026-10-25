// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：piece.h
// 描述：棋子模型
// 创建日期：2026-02-21
// 作者：SpaceGo Team
// ============================================================================

#pragma once
#include <string>

namespace SpaceGo {
namespace GUI {
namespace Models {

// 棋子颜色
enum class PieceColor {
    None,
    Black,
    White
};

class Piece {
public:
    PieceColor color;
    int x;
    int y;
    int moveNumber;

    Piece();
    Piece(PieceColor c, int px, int py, int num = 0);

    // 获取颜色名称
    std::wstring GetColorName() const;

    // 是否为空
    bool IsEmpty() const;
};

} // namespace Models
} // namespace GUI
} // namespace SpaceGo
