// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：piece.cpp
// 描述：棋子模型实现
// 创建日期：2026-02-21
// 作者：SpaceGo Team
// ============================================================================

#include "models/piece.h"

namespace SpaceGo {
namespace GUI {
namespace Models {

Piece::Piece()
    : color(PieceColor::None), x(-1), y(-1), moveNumber(0) {
}

Piece::Piece(PieceColor c, int px, int py, int num)
    : color(c), x(px), y(py), moveNumber(num) {
}

std::wstring Piece::GetColorName() const {
    switch (color) {
        case PieceColor::Black:
            return L"黑方";
        case PieceColor::White:
            return L"白方";
        default:
            return L"空";
    }
}

bool Piece::IsEmpty() const {
    return color == PieceColor::None;
}

} // namespace Models
} // namespace GUI
} // namespace SpaceGo
