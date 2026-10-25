// ============================================================================
// SpaceGo 维空弈 - GUI系统
// 文件名：game.cpp
// 描述：游戏模型实现
// 创建日期：2026-02-21
// 作者：SpaceGo Team
// ============================================================================

#include "models/game.h"

namespace SpaceGo {
namespace GUI {
namespace Models {

Game::Game()
    : state(GameState::Idle), mode(GameMode::PvP), board(19), 
      currentPlayer(PieceColor::Black), moveCount(0) {
}

Game::~Game() {
}

void Game::StartNewGame(GameMode gm, int boardSize) {
    mode = gm;
    board.SetSize(boardSize);
    currentPlayer = PieceColor::Black;
    moveCount = 0;
    moveHistory.clear();
    state = GameState::Playing;
}

bool Game::MakeMove(int x, int y) {
    if (state != GameState::Playing) {
        return false;
    }

    if (board.PlacePiece(x, y, currentPlayer, moveCount + 1)) {
        moveCount++;
        MoveRecord record = {x, y, currentPlayer, moveCount};
        moveHistory.push_back(record);
        SwitchPlayer();
        return true;
    }
    return false;
}

bool Game::UndoMove() {
    if (moveHistory.empty() || state != GameState::Playing) {
        return false;
    }

    MoveRecord lastMove = moveHistory.back();
    board.RemovePiece(lastMove.x, lastMove.y);
    moveHistory.pop_back();
    moveCount--;
    currentPlayer = lastMove.color;
    return true;
}

void Game::Pass() {
    if (state != GameState::Playing) {
        return;
    }
    SwitchPlayer();
}

void Game::Pause() {
    if (state == GameState::Playing) {
        state = GameState::Paused;
    }
}

void Game::Resume() {
    if (state == GameState::Paused) {
        state = GameState::Playing;
    }
}

void Game::Finish() {
    state = GameState::Finished;
}

void Game::Reset() {
    board.Reset();
    currentPlayer = PieceColor::Black;
    moveCount = 0;
    moveHistory.clear();
    state = GameState::Idle;
}

PieceColor Game::GetCurrentPlayer() const {
    return currentPlayer;
}

void Game::SwitchPlayer() {
    if (currentPlayer == PieceColor::Black) {
        currentPlayer = PieceColor::White;
    } else {
        currentPlayer = PieceColor::Black;
    }
}

GameState Game::GetState() const {
    return state;
}

const std::vector<MoveRecord>& Game::GetMoveHistory() const {
    return moveHistory;
}

} // namespace Models
} // namespace GUI
} // namespace SpaceGo
