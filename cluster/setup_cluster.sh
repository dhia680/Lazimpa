#!/bin/bash
# Interactive setup script for LazImpa cluster experiments

echo "========================================================================"
echo "LazImpa Cluster Setup"
echo "========================================================================"
echo ""

# Get user information
read -p "Enter your NetID (e.g., aquispe): " NETID
read -p "Enter your email (e.g., anzony.quispe@gmail.com): " EMAIL
read -p "Enter full path to Lazimpa on cluster (default: \$HOME/Lazimpa): " LAZIMPA_PATH
LAZIMPA_PATH=${LAZIMPA_PATH:-\$HOME/Lazimpa}

echo ""
echo "Checking available modules..."
echo ""

# Check for Python
echo "Available Python modules:"
module avail python 2>&1 | grep -i python | head -5
read -p "Enter Python module name (e.g., python/3.9): " PYTHON_MODULE

# Check for CUDA
echo ""
echo "Available CUDA modules:"
module avail cuda 2>&1 | grep -i cuda | head -5
read -p "Enter CUDA module name (e.g., cuda/11.8): " CUDA_MODULE

# Check for PyTorch
echo ""
echo "Available PyTorch modules (leave blank if none):"
module avail pytorch 2>&1 | grep -i pytorch | head -5
read -p "Enter PyTorch module name (or press Enter to skip): " PYTORCH_MODULE

echo ""
echo "========================================================================"
echo "Configuration Summary"
echo "========================================================================"
echo "NetID: $NETID"
echo "Email: $EMAIL"
echo "Path: $LAZIMPA_PATH"
echo "Python: $PYTHON_MODULE"
echo "CUDA: $CUDA_MODULE"
echo "PyTorch: $PYTORCH_MODULE"
echo ""
read -p "Update all scripts with this configuration? [y/N]: " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup canceled."
    exit 1
fi

echo ""
echo "Updating scripts..."

# Update email in all scripts
for script in *.sh; do
    if [ -f "$script" ]; then
        sed -i.bak "s/anzony.quispe@gmail.com/$EMAIL/g" "$script"
        echo "  Updated email in $script"
    fi
done

# Update NetID in README files
for readme in *.md; do
    if [ -f "$readme" ]; then
        sed -i.bak "s/netid@nd.edu/$NETID@nd.edu/g" "$readme"
        sed -i.bak "s/netid@crc.nd.edu/$NETID@crc.nd.edu/g" "$readme"
        echo "  Updated NetID in $readme"
    fi
done

# Update module loads
for script in *.sh; do
    if [ -f "$script" ] && [ "$script" != "setup_cluster.sh" ] && [ "$script" != "manage_jobs.sh" ]; then
        # Update Python module
        sed -i.bak "s/module load python\/[0-9.]\+/module load $PYTHON_MODULE/g" "$script"

        # Update CUDA module
        sed -i.bak "s/module load cuda\/[0-9.]\+/module load $CUDA_MODULE/g" "$script"

        # Update or remove PyTorch module
        if [ -n "$PYTORCH_MODULE" ]; then
            # Add PyTorch module if it doesn't exist
            if ! grep -q "module load pytorch" "$script"; then
                sed -i.bak "/module load cuda/a module load $PYTORCH_MODULE" "$script"
            else
                sed -i.bak "s/module load pytorch\/[0-9.]\+/module load $PYTORCH_MODULE/g" "$script"
            fi
        else
            # Remove PyTorch module line and add comment
            sed -i.bak "/module load pytorch/d" "$script"
        fi

        echo "  Updated modules in $script"
    fi
done

# Update path
for script in *.sh; do
    if [ -f "$script" ] && [ "$script" != "setup_cluster.sh" ]; then
        sed -i.bak "s|\$HOME/Lazimpa|$LAZIMPA_PATH|g" "$script"
        echo "  Updated path in $script"
    fi
done

# Clean up backup files
rm -f *.bak

echo ""
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Upload to cluster:"
echo "   rsync -avz --exclude 'results/' . $NETID@crc.nd.edu:$LAZIMPA_PATH/"
echo ""
echo "2. SSH to cluster:"
echo "   ssh $NETID@crc.nd.edu"
echo ""
echo "3. Test environment:"
echo "   cd $LAZIMPA_PATH/cluster"
echo "   module load $PYTHON_MODULE $CUDA_MODULE"
if [ -n "$PYTORCH_MODULE" ]; then
    echo "   module load $PYTORCH_MODULE"
fi
echo "   python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "4. Submit test job:"
echo "   qsub test_single.sh"
echo ""
echo "See QUICKSTART.md for detailed instructions."
echo ""
