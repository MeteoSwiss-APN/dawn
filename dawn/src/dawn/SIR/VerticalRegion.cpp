#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTStringifier.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/VerticalRegion.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/StringUtil.h"
#include <sstream>

namespace dawn {

CompareResult sir::VerticalRegion::comparison(const sir::VerticalRegion& rhs) const {
  std::string output;
  if(LoopOrder != rhs.LoopOrder) {
    output += dawn::format("[VerticalRegion mismatch] Loop order does not match\n"
                           "  Actual:\n"
                           "    %s\n"
                           "  Expected:\n"
                           "    %s",
                           static_cast<int>(LoopOrder), static_cast<int>(rhs.LoopOrder));
    return CompareResult{output, false};
  }

  auto intervalComp = VerticalInterval->comparison(*(rhs.VerticalInterval));
  if(!static_cast<bool>(intervalComp)) {
    output += "[VerticalRegion mismatch] Intervals do not match\n";
    output += intervalComp.why();
    return CompareResult{output, false};
  } else if(IterationSpace[0] != rhs.IterationSpace[0]) {
    output += "[VerticalRegion mismatch] iteration space in i do not match\n";
    return CompareResult{output, false};
  } else if(IterationSpace[1] != rhs.IterationSpace[1]) {
    output += "[VerticalRegion mismatch] iteration space in j do not match\n";
    return CompareResult{output, false};
  }

  auto astComp = compareAst(Ast, rhs.Ast);
  if(!astComp.second) {
    output += "[VerticalRegion mismatch] ASTs do not match\n";
    output += astComp.first;
    return CompareResult{output, false};
  } else {
    return CompareResult{output, true};
  }
}

bool sir::VerticalRegion::operator==(const sir::VerticalRegion& rhs) const {
  // casted to bool by return statement
  return this->comparison(rhs);
}

std::shared_ptr<sir::VerticalRegion> sir::VerticalRegion::clone() const {
  auto retval =
      std::make_shared<sir::VerticalRegion>(Ast->clone(), VerticalInterval, LoopOrder, Loc);
  retval->IterationSpace = IterationSpace;
  return retval;
}

namespace {
/// @brief Allow direct comparison of the Stmts of an AST
class DiffWriter final : public ast::ASTVisitorForwarding {
public:
  virtual void visit(const std::shared_ptr<ast::VerticalRegionDeclStmt>& stmt) override {
    statements_.push_back(stmt);
    stmt->getVerticalRegion()->Ast->getRoot()->accept(*this);
  }

  virtual void visit(const std::shared_ptr<ast::ReturnStmt>& stmt) override {
    statements_.push_back(stmt);
    ast::ASTVisitorForwarding::visit(stmt);
  }

  virtual void visit(const std::shared_ptr<ast::ExprStmt>& stmt) override {
    statements_.push_back(stmt);
    ast::ASTVisitorForwarding::visit(stmt);
  }

  virtual void visit(const std::shared_ptr<ast::BlockStmt>& stmt) override {
    statements_.push_back(stmt);
    ast::ASTVisitorForwarding::visit(stmt);
  }

  virtual void visit(const std::shared_ptr<ast::VarDeclStmt>& stmt) override {
    statements_.push_back(stmt);
    ast::ASTVisitorForwarding::visit(stmt);
  }

  virtual void visit(const std::shared_ptr<ast::IfStmt>& stmt) override {
    statements_.push_back(stmt);
    ast::ASTVisitorForwarding::visit(stmt);
  }

  std::vector<std::shared_ptr<ast::Stmt>> getStatements() const { return statements_; }

  std::pair<std::string, bool> compare(const DiffWriter& other) {

    std::size_t minSize = std::min(statements_.size(), other.getStatements().size());
    if(minSize == 0 && (statements_.size() != other.getStatements().size()))
      return std::make_pair("[AST mismatch] AST is empty", false);

    for(std::size_t idx = 0; idx < minSize; ++idx) {
      if(!statements_[idx]->equals(other.getStatements()[idx].get())) {
        return std::make_pair(
            dawn::format("[AST mismatch] Statement mismatch\n"
                         "  Actual:\n"
                         "    %s\n"
                         "  Expected:\n"
                         "    %s",
                         indent(ast::ASTStringifier::toString(statements_[idx]), 4),
                         indent(ast::ASTStringifier::toString(other.getStatements()[idx]), 4)),
            false);
      }
    }

    return std::make_pair("", true);
  }

private:
  std::vector<std::shared_ptr<ast::Stmt>> statements_;
};

} // namespace

namespace sir {

/// @brief Compares two ASTs
std::pair<std::string, bool> compareAst(const std::shared_ptr<ast::AST>& lhs,
                                        const std::shared_ptr<ast::AST>& rhs) {
  if(lhs->getRoot()->equals(rhs->getRoot().get()))
    return std::make_pair("", true);

  DiffWriter lhsDW, rhsDW;
  lhs->accept(lhsDW);
  rhs->accept(rhsDW);

  auto comp = lhsDW.compare(rhsDW);
  if(!comp.second)
    return comp;

  return std::make_pair("", true);
}

} // namespace sir

namespace ast {

//===------------------------------------------------------------------------------------------===//
//     VerticalRegionDeclStmt
//===------------------------------------------------------------------------------------------===//

VerticalRegionDeclStmt::VerticalRegionDeclStmt(
    std::unique_ptr<StmtData> data, const std::shared_ptr<sir::VerticalRegion>& verticalRegion,
    SourceLocation loc)
    : Stmt(std::move(data), Kind::VerticalRegionDeclStmt, loc), verticalRegion_(verticalRegion) {
  DAWN_ASSERT_MSG((checkSameDataType(*verticalRegion_->Ast->getRoot())),
                  "Trying to insert vertical region with different data type");
}

VerticalRegionDeclStmt::VerticalRegionDeclStmt(const VerticalRegionDeclStmt& stmt)
    : Stmt(stmt), verticalRegion_(stmt.getVerticalRegion()->clone()) {}

VerticalRegionDeclStmt& VerticalRegionDeclStmt::operator=(VerticalRegionDeclStmt stmt) {
  assign(stmt);
  verticalRegion_ = std::move(stmt.getVerticalRegion());
  return *this;
}

VerticalRegionDeclStmt::~VerticalRegionDeclStmt() {}

std::shared_ptr<Stmt> VerticalRegionDeclStmt::clone() const {
  return std::make_shared<VerticalRegionDeclStmt>(*this);
}

bool VerticalRegionDeclStmt::equals(const Stmt* other, bool compareData) const {
  const VerticalRegionDeclStmt* otherPtr = dyn_cast<VerticalRegionDeclStmt>(other);
  return otherPtr && Stmt::equals(other, compareData) &&
         *(verticalRegion_.get()) == *(otherPtr->verticalRegion_.get());
}

} // namespace ast
} // namespace dawn
