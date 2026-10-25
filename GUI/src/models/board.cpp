// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：board.cpp
// 描述：棋盘模型实现
// 创建日期：2026-02-21
// 作者：SpaceGo Team
// ============================================================================

#include "models/board.h"

namespace SpaceGo {
namespace GUI {
namespace Models {

Board::Board(int size)
    : gridSize(size) {
    Reset();
}

Board::~Board() {
}

void Board::Reset() {
    pieces.resize(gridSize);
    for (int i = 0; i < gridSize; ++i) {
        pieces[i].resize(gridSize);
        for (int j = 0; j < gridSize; ++j) {
            pieces[i][j] = Piece();
        }
    }
}

void Board::SetSize(int size) {
    gridSize = size;
    Reset();
}

bool Board::PlacePiece(int x, int y, PieceColor color, int moveNumber) {
    if (!IsValidPosition(x, y) || !IsEmpty(x, y)) {
        return false;
    }
    pieces[y][x] = Piece(color, x, y, moveNumber);
    return true;
}

void Board::RemovePiece(int x, int y) {
    if (IsValidPosition(x, y)) {
        pieces[y][x] = Piece();
    }
}

Piece Board::GetPiece(int x, int y) const {
    if (IsValidPosition(x, y)) {
        return pieces[y][x];
    }
    return Piece();
}

bool Board::IsValidPosition(int x, int y) const {
    return x >= 0 && x < gridSize && y >= 0 && y < gridSize;
}

bool Board::IsEmpty(int x, int y) const {
    if (!IsValidPosition(x, y)) {
        return false;
    }
    return pieces[y][x].IsEmpty();
}

int Board::GetPieceCount() const {
    int count = 0;
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            if (!pieces[i][j].IsEmpty()) {
                count++;
            }
        }
    }
    return count;
}

} // namespace Models
} // namespace GUI
} // namespace SpaceGo
