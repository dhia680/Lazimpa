#!/bin/bash
# Verify that test job completed successfully
# Run this AFTER test_single.sh finishes

echo "========================================================================"
echo "LazImpa Test Verification"
echo "========================================================================"
echo ""

# Find most recent test output
TEST_OUT=$(ls -t lazimpa_test.o* 2>/dev/null | head -1)
TEST_ERR=$(ls -t lazimpa_test.e* 2>/dev/null | head -1)

if [ -z "$TEST_OUT" ]; then
    echo "❌ ERROR: No test output files found"
    echo "   Have you run: qsub test_single.sh ?"
    exit 1
fi

echo "Checking test output: $TEST_OUT"
echo ""

PASS=0
FAIL=0

# Check 1: Job finished
if grep -q "Job finished at" "$TEST_OUT"; then
    echo "✓ Test job completed"
    ((PASS++))
else
    echo "✗ Test job did not finish"
    ((FAIL++))
fi

# Check 2: GPU detected
if grep -q "CUDA_VISIBLE_DEVICES" "$TEST_OUT"; then
    echo "✓ GPU environment set"
    ((PASS++))
else
    echo "✗ GPU not detected"
    ((FAIL++))
fi

# Check 3: Training ran
EPOCHS=$(grep -c "^Epoch:" "$TEST_OUT")
if [ "$EPOCHS" -ge 10 ]; then
    echo "✓ Training completed ($EPOCHS epochs)"
    ((PASS++))
else
    echo "✗ Training incomplete (only $EPOCHS epochs)"
    ((FAIL++))
fi

# Check 4: Messages generated
if grep -q "message:" "$TEST_OUT"; then
    echo "✓ Messages generated"
    ((PASS++))
else
    echo "✗ No messages found"
    ((FAIL++))
fi

# Check 5: Impatient receiver working
if grep -q "Impatient score=" "$TEST_OUT"; then
    echo "✓ Impatient receiver working"
    ((PASS++))
else
    echo "✗ Impatient receiver not working"
    ((FAIL++))
fi

# Check 6: No major errors
if [ -f "$TEST_ERR" ]; then
    ERROR_LINES=$(grep -i "error\|exception\|traceback" "$TEST_ERR" | wc -l)
    if [ "$ERROR_LINES" -eq 0 ]; then
        echo "✓ No errors in error log"
        ((PASS++))
    else
        echo "✗ Found $ERROR_LINES error messages"
        ((FAIL++))
    fi
else
    echo "✓ No error file (good sign)"
    ((PASS++))
fi

# Check 7: Output files created
if [ -d "$HOME/Lazimpa/cluster_test/transformer_lazimpa/messages" ]; then
    MSG_COUNT=$(ls $HOME/Lazimpa/cluster_test/transformer_lazimpa/messages/*.npy 2>/dev/null | wc -l)
    if [ "$MSG_COUNT" -gt 0 ]; then
        echo "✓ Message files saved ($MSG_COUNT files)"
        ((PASS++))
    else
        echo "✗ No message files saved"
        ((FAIL++))
    fi
else
    echo "✗ Output directory not created"
    ((FAIL++))
fi

echo ""
echo "========================================================================"
echo "VERIFICATION RESULTS"
echo "========================================================================"
echo "Passed: $PASS/7"
echo "Failed: $FAIL/7"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
    echo ""
    echo "Your cluster setup is working correctly."
    echo "You can now run the full experiments:"
    echo ""
    echo "  qsub run_parallel_array.sh    # Fast (2-4 hours)"
    echo "  OR"
    echo "  qsub run_all_experiments.sh   # Slow (40-80 hours)"
    echo ""
    exit 0
else
    echo "⚠️  SOME TESTS FAILED"
    echo ""
    echo "Please review the issues above before running full experiments."
    echo ""
    echo "Common fixes:"
    echo "  - Check module names: module avail python"
    echo "  - Install PyTorch: pip install --user torch"
    echo "  - Check GPU queue: qstat -F gpu_card"
    echo ""
    echo "View full output:"
    echo "  cat $TEST_OUT"
    echo "  cat $TEST_ERR"
    echo ""
    exit 1
fi
