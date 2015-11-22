#include <vector>
#include <stdlib.h>
#include <string.h>
#include <iostream>     // std::ios, std::istream, std::cout
#include <fstream>      // std::filebuf

int main(int argc, char *argv[])
{
    if (argc != 13)
    {
        std::cout << "usage: ./meshgen nx ny min_x min_y max_x max_y bcid factor uniform ul_lr tblra filename\n";
        return -1;
    }
    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    double min_x = atof(argv[3]);
    double min_y = atof(argv[4]);
    double max_x = atof(argv[5]);
    double max_y = atof(argv[6]);
    int bcid = atoi(argv[7]);
    double factor = atof(argv[8]);
    bool uniform = (atoi(argv[9])==0?false:true);
    bool ul_lr = atoi(argv[10])==1? true : false; // upper left to lower right diagonal, or ur_ll?
    char* borders = argv[11];
    char* fname = argv[12];

    bool bleft = false;
    if (strchr(borders,'l') != NULL)
        bleft = true;
    bool bright = false;
    if (strchr(borders,'r') != NULL)
        bright = true;
    bool btop = false;
    if (strchr(borders,'t') != NULL)
        btop = true;
    bool bbottom = false;
    if (strchr(borders,'b') != NULL)
        bbottom = true;
    if (strchr(borders,'a') != NULL)
    {
        bleft = bright = btop = bbottom = true;
    }

    int n_tri = nx*ny*2;
    int n_nodes = (nx+1)*(ny+1);

    std::vector<std::vector<double> > nodes;
    std::vector<std::vector<int> > tri;

    double fracx = (max_x-min_x)/(double)nx;
    double fracy = (max_y-min_y)/(double)ny;
    for (int y = 0; y <= ny; y++)
    {
        std::vector<double> node(3,0.0);
        node[1] = min_y + y*fracy;
        for (int x = 0; x <= nx; x++)
        {
            node[0] = (min_x + x*fracx);
            nodes.push_back(node);
        }
    }

    for (int y = 0; y < ny; y++)
    {
        std::vector<int> tri1(3);
        std::vector<int> tri2(3);
        for (int x = 0; x < nx; x++)
        {
            int n_id = x + y*(nx+1);
            if (ul_lr)
            {
                tri1[0] = n_id;
                tri1[1] = n_id + 1;
                tri1[2] = n_id + (nx+1);

                tri2[0] = n_id + 1;
                tri2[1] = n_id + (nx+1) + 1;
                tri2[2] = n_id + (nx+1);
            }
            else
            {
                tri1[0] = n_id;
                tri1[1] = n_id + (nx+1) + 1;
                tri1[2] = n_id + 1;

                tri2[0] = n_id + (nx+1) + 1;
                tri2[1] = n_id;
                tri2[2] = n_id + (nx+1);
            }
            tri.push_back(tri1);
            tri.push_back(tri2);
        }
    }

    std::string meshname = fname;
    meshname += ".xda";
    std::string forcename = fname;
    forcename += "_f";
    std::filebuf fb;
    fb.open (meshname.c_str(), std::ios::out);
    std::ostream os(&fb);

    os << "libMesh-0.7.0+\n";
    os << n_tri << "        # number of elements\n";
    os << n_nodes << "      # number of nodes\n";
    os << ".        # boundary condition specification file\n";
    os << "n/a      # subdomain id specification file\n";
    os << "n/a      # processor id specification file\n";
    os << "n/a      # p-level specification file\n";
    os << n_tri << "      # n_elem at level 0, [ type (n0 ... nN-1) ]\n";
    for (unsigned int i = 0; i < tri.size(); i++)
    {
        std::vector<int> cur_tri = tri[i];
        os << "3 " << cur_tri[0] << " " << cur_tri[1] << " " << cur_tri[2] << "\n";
    }
    for (unsigned int i = 0; i < nodes.size(); i++)
    {
        std::vector<double> cur_node = nodes[i];
        os << cur_node[0] << " " << cur_node[1] << " " << cur_node[2] << "\n";
    }
    os << (2*nx+2*ny) << "        # number of boundary conditions\n";
    for (int i = 0; i < nx; i++)
    {
        if (ul_lr)
        {
            if (bbottom)
                os << 2*i << " 0 " << bcid << "\n"; // unterer Rand
            if (btop)
                os << 2*nx*ny-2*i-1 << " 1 " << bcid << "\n"; // oberer Rand
        }
        else
        {
            if (bbottom)
                os << 2*i << " 2 " << bcid << "\n"; // unterer Rand
            if (btop)
                os << 2*nx*ny-2*i-1 << " 2 " << bcid << "\n"; // oberer Rand
        }
    }
    for (int i = 0; i < ny; i++)
    {
        if (ul_lr)
        {
            if (bleft)
                os << 2*nx*i << " 2 " << bcid << "\n"; // linker Rand
            if (bright)
                os << 2*nx*(i+1)-1 << " 0 " << bcid << "\n"; // rechter Rand
        }
        else
        {
            if (bleft)
                os << 2*nx*i+1 << " 1 " << bcid << "\n"; // linker Rand
            if (bright)
                os << 2*nx*(i+1)-2 << " 1 " << bcid << "\n"; // rechter Rand
        }
    }
    fb.close();

    fb.open (forcename.c_str(), std::ios::out);
    std::ostream os2(&fb);

    os2 << n_nodes << "\n";
    if (uniform)
    {
        os2 << (factor*(max_x/(double)nx)*(max_y/(double)ny)) << "\n";
        for (unsigned int i = 0; i < nodes.size()-1; i++)
            os2 << "0 0 1 0 0 0\n";
    }
    else
    {
        os2 << factor << "\n";
        for (unsigned int i = 0; i < nodes.size()-1; i++)
        {
            if (i == n_nodes/2)
                os2 << "0 0 1 0 0 0\n";
            else
                os2 << "0 0 0 0 0 0\n";
        }
    }
    fb.close();

    return 0;
}
