//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "dawn/Support/ArrayRef.h"
#include "dawn/Support/StringUtil.h"

#include <gtest/gtest.h>
#include <ostream>
#include <vector>

namespace dawn {

template <class T>
std::ostream& operator<<(std::ostream& os, const ArrayRef<T>& a) {
  return os << RangeToString()(a);
}

} // namespace dawn

using namespace dawn;

// Check that the ArrayRef-of-pointer converting constructor only allows adding
// cv qualifiers (not removing them, or otherwise changing the type)
static_assert(std::is_convertible<ArrayRef<int*>, ArrayRef<const int*>>::value, "Adding const");
static_assert(std::is_convertible<ArrayRef<int*>, ArrayRef<volatile int*>>::value,
              "Adding volatile");
static_assert(!std::is_convertible<ArrayRef<int*>, ArrayRef<float*>>::value,
              "Changing pointer of one type to a pointer of another");
static_assert(!std::is_convertible<ArrayRef<const int*>, ArrayRef<int*>>::value, "Removing const");
static_assert(!std::is_convertible<ArrayRef<volatile int*>, ArrayRef<int*>>::value,
              "Removing volatile");

// Check that we can't accidentally assign a temporary location to an ArrayRef.
// (Unfortunately we can't make use of the same thing with constructors.)
//
// Disable this check under MSVC; even MSVC 2015 isn't consistent between std::is_assignable and
// actually writing such an assignment.
static_assert(!std::is_assignable<ArrayRef<int*>, int*>::value,
              "Assigning from single prvalue element");
static_assert(!std::is_assignable<ArrayRef<int*>, int*&&>::value,
              "Assigning from single xvalue element");
static_assert(std::is_assignable<ArrayRef<int*>, int*&>::value,
              "Assigning from single lvalue element");
static_assert(!std::is_assignable<ArrayRef<int*>, std::initializer_list<int*>>::value,
              "Assigning from an initializer list");

namespace {

TEST(ArrayRefTest, DropBack) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(TheNumbers, AR1.size() - 1);
  EXPECT_TRUE(AR1.drop_back().equals(AR2));

  // Check that drop_back accepts size_t-sized numbers.
  ArrayRef<char> AR3((const char*)0x10000, SIZE_MAX - 0x10000);
  EXPECT_EQ(1U, AR3.drop_back(AR3.size() - 1).size());
}

TEST(ArrayRefTest, DropFront) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(&TheNumbers[2], AR1.size() - 2);
  EXPECT_TRUE(AR1.drop_front(2).equals(AR2));

  // Check that drop_front accepts size_t-sized numbers.
  ArrayRef<char> AR3((const char*)0x10000, SIZE_MAX - 0x10000);
  EXPECT_EQ(1U, AR3.drop_front(AR3.size() - 1).size());
}

TEST(ArrayRefTest, DropWhile) {
  static const int TheNumbers[] = {1, 3, 5, 8, 10, 11};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> Expected = AR1.drop_front(3);
  EXPECT_EQ(Expected, AR1.drop_while([](const int& N) { return N % 2 == 1; }));

  EXPECT_EQ(AR1, AR1.drop_while([](const int& N) { return N < 0; }));
  EXPECT_EQ(ArrayRef<int>(), AR1.drop_while([](const int& N) { return N > 0; }));
}

TEST(ArrayRefTest, DropUntil) {
  static const int TheNumbers[] = {1, 3, 5, 8, 10, 11};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> Expected = AR1.drop_front(3);
  EXPECT_EQ(Expected, AR1.drop_until([](const int& N) { return N % 2 == 0; }));

  EXPECT_EQ(ArrayRef<int>(), AR1.drop_until([](const int& N) { return N < 0; }));
  EXPECT_EQ(AR1, AR1.drop_until([](const int& N) { return N > 0; }));
}

TEST(ArrayRefTest, TakeBack) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(AR1.end() - 1, 1);
  EXPECT_TRUE(AR1.take_back().equals(AR2));
}

TEST(ArrayRefTest, TakeFront) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(AR1.data(), 2);
  EXPECT_TRUE(AR1.take_front(2).equals(AR2));
}

TEST(ArrayRefTest, TakeWhile) {
  static const int TheNumbers[] = {1, 3, 5, 8, 10, 11};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> Expected = AR1.take_front(3);
  EXPECT_EQ(Expected, AR1.take_while([](const int& N) { return N % 2 == 1; }));

  EXPECT_EQ(ArrayRef<int>(), AR1.take_while([](const int& N) { return N < 0; }));
  EXPECT_EQ(AR1, AR1.take_while([](const int& N) { return N > 0; }));
}

TEST(ArrayRefTest, TakeUntil) {
  static const int TheNumbers[] = {1, 3, 5, 8, 10, 11};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> Expected = AR1.take_front(3);
  EXPECT_EQ(Expected, AR1.take_until([](const int& N) { return N % 2 == 0; }));

  EXPECT_EQ(AR1, AR1.take_until([](const int& N) { return N < 0; }));
  EXPECT_EQ(ArrayRef<int>(), AR1.take_until([](const int& N) { return N > 0; }));
}

TEST(ArrayRefTest, Equals) {
  static const int A1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  ArrayRef<int> AR1(A1);
  EXPECT_TRUE(AR1.equals({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_FALSE(AR1.equals({8, 1, 2, 4, 5, 6, 6, 7}));
  EXPECT_FALSE(AR1.equals({2, 4, 5, 6, 6, 7, 8, 1}));
  EXPECT_FALSE(AR1.equals({0, 1, 2, 4, 5, 6, 6, 7}));
  EXPECT_FALSE(AR1.equals({1, 2, 42, 4, 5, 6, 7, 8}));
  EXPECT_FALSE(AR1.equals({42, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_FALSE(AR1.equals({1, 2, 3, 4, 5, 6, 7, 42}));
  EXPECT_FALSE(AR1.equals({1, 2, 3, 4, 5, 6, 7}));
  EXPECT_FALSE(AR1.equals({1, 2, 3, 4, 5, 6, 7, 8, 9}));

  ArrayRef<int> AR1a = AR1.drop_back();
  EXPECT_TRUE(AR1a.equals({1, 2, 3, 4, 5, 6, 7}));
  EXPECT_FALSE(AR1a.equals({1, 2, 3, 4, 5, 6, 7, 8}));

  ArrayRef<int> AR1b = AR1a.slice(2, 4);
  EXPECT_TRUE(AR1b.equals({3, 4, 5, 6}));
  EXPECT_FALSE(AR1b.equals({2, 3, 4, 5, 6}));
  EXPECT_FALSE(AR1b.equals({3, 4, 5, 6, 7}));
}

TEST(ArrayRefTest, EmptyEquals) { EXPECT_TRUE(ArrayRef<unsigned>() == ArrayRef<unsigned>()); }

TEST(ArrayRefTest, Slice) {
  // Check that slice accepts size_t-sized numbers.
  ArrayRef<char> AR((const char*)0x10000, SIZE_MAX - 0x10000);
  EXPECT_EQ(1U, AR.slice(AR.size() - 1).size());
  EXPECT_EQ(AR.size() - 1, AR.slice(1, AR.size() - 1).size());
}

TEST(ArrayRefTest, ConstConvert) {
  int buf[4];
  for(int i = 0; i < 4; ++i)
    buf[i] = i;

  static int* A[] = {&buf[0], &buf[1], &buf[2], &buf[3]};
  ArrayRef<const int*> a((ArrayRef<int*>(A)));
  a = ArrayRef<int*>(A);
}

static std::vector<int> ReturnTest12() { return {1, 2}; }
static void ArgTest12(ArrayRef<int> A) {
  EXPECT_EQ(2U, A.size());
  EXPECT_EQ(1, A[0]);
  EXPECT_EQ(2, A[1]);
}

TEST(ArrayRefTest, InitializerList) {
  std::initializer_list<int> init_list = {0, 1, 2, 3, 4};
  ArrayRef<int> A = init_list;
  for(int i = 0; i < 5; ++i)
    EXPECT_EQ(i, A[i]);

  std::vector<int> B = ReturnTest12();
  A = B;
  EXPECT_EQ(1, A[0]);
  EXPECT_EQ(2, A[1]);

  ArgTest12({1, 2});
}

TEST(ArrayRefTest, EmptyInitializerList) {
  ArrayRef<int> A = {};
  EXPECT_TRUE(A.empty());

  A = {};
  EXPECT_TRUE(A.empty());
}

// Test that makeArrayRef works on ArrayRef (no-op)
TEST(ArrayRefTest, makeArrayRef) {
  static const int A1[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // No copy expected for non-const ArrayRef (true no-op)
  ArrayRef<int> AR1(A1);
  ArrayRef<int>& AR1Ref = makeArrayRef(AR1);
  EXPECT_EQ(&AR1, &AR1Ref);

  // A copy is expected for non-const ArrayRef (thin copy)
  const ArrayRef<int> AR2(A1);
  const ArrayRef<int>& AR2Ref = makeArrayRef(AR2);
  EXPECT_NE(&AR2Ref, &AR2);
  EXPECT_TRUE(AR2.equals(AR2Ref));
}

} // anonymous namespace
